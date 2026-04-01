"""Full pipeline benchmark: DeepFilterNet3 vs DPDFNet as Stage 1 denoiser.

Tests the COMPLETE pipeline (denoiser → ClearVoice → Pedalboard → LUFS → Limiter)
with each denoiser, so we compare final output quality, not isolated denoiser quality.

Also re-runs DeepFilterNet3 correctly at its native 48kHz sample rate.
"""

import types
import sys
import torchaudio

if not hasattr(torchaudio, "backend"):
    backend_module = types.ModuleType("torchaudio.backend")
    common_module = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0, bits_per_sample=0, encoding=""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    common_module.AudioMetaData = AudioMetaData
    backend_module.common = common_module
    sys.modules["torchaudio.backend"] = backend_module
    sys.modules["torchaudio.backend.common"] = common_module
    torchaudio.backend = backend_module

import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from clearvoice import ClearVoice
from df.enhance import enhance as df_enhance
from df.enhance import init_df
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    PeakFilter,
    Pedalboard,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_M4A = Path("recording.m4a")
OUTPUT_DIR = Path("benchmark_outputs")
WAV_48K = OUTPUT_DIR / "input_48k.wav"

OUTPUT_SR = 48000
LUFS_TARGET = -18.0
LIMITER_CEILING_DB = -1.5


def ensure_input():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not WAV_48K.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(INPUT_M4A), "-ar", "48000", "-ac", "1", str(WAV_48K)],
            capture_output=True, check=True,
        )


def build_mastering_chain():
    mastering = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        PeakFilter(cutoff_frequency_hz=300, gain_db=-3.0, q=1.0),
        Compressor(threshold_db=-20, ratio=2.0, attack_ms=15, release_ms=100),
        Compressor(threshold_db=-15, ratio=3.0, attack_ms=10, release_ms=80),
        PeakFilter(cutoff_frequency_hz=6000, gain_db=-4.0, q=2.0),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5, q=0.8),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0, q=0.7),
    ])
    limiter = Pedalboard([Limiter(threshold_db=LIMITER_CEILING_DB)])
    return mastering, limiter


def run_stages_2_to_5(audio_np: np.ndarray, sr: int, cv: ClearVoice, mastering, limiter, label: str) -> tuple[np.ndarray, dict]:
    """Run ClearVoice → Pedalboard → LUFS → Limiter on denoised audio."""
    timings = {}

    # Stage 2: ClearVoice MossFormer2 (numpy mode)
    if audio_np.ndim == 1:
        cv_input = audio_np[np.newaxis, :].astype(np.float32)
    else:
        cv_input = audio_np.astype(np.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        result = cv(cv_input)
    timings["stage2_clearvoice_s"] = round(time.perf_counter() - t0, 3)

    if isinstance(result, dict):
        enhanced = list(result.values())[0]
    elif isinstance(result, torch.Tensor):
        enhanced = result.numpy()
    else:
        enhanced = np.array(result, dtype=np.float32)
    enhanced = np.ascontiguousarray(enhanced.astype(np.float32, copy=False))
    if enhanced.ndim == 1:
        enhanced = enhanced[np.newaxis, :]
    logger.info("[%s] Stage 2 done (ClearVoice): %.2fs", label, timings["stage2_clearvoice_s"])

    # Stage 3: Pedalboard DSP
    t0 = time.perf_counter()
    mastered = mastering(enhanced, sample_rate=OUTPUT_SR, reset=True)
    timings["stage3_mastering_s"] = round(time.perf_counter() - t0, 3)
    logger.info("[%s] Stage 3 done (mastering): %.2fs", label, timings["stage3_mastering_s"])

    # Stage 4: LUFS normalization
    t0 = time.perf_counter()
    mono = np.ascontiguousarray(mastered[0].astype(np.float64, copy=False))
    meter = pyln.Meter(OUTPUT_SR)
    loudness = meter.integrated_loudness(mono)
    if np.isfinite(loudness):
        normalized = pyln.normalize.loudness(mono, loudness, LUFS_TARGET).astype(np.float32)
    else:
        normalized = mono.astype(np.float32)
    timings["stage4_lufs_s"] = round(time.perf_counter() - t0, 3)
    logger.info("[%s] Stage 4 done (LUFS %.1f → %.1f): %.2fs", label, loudness, LUFS_TARGET, timings["stage4_lufs_s"])

    # Stage 5: Limiter
    t0 = time.perf_counter()
    limited = limiter(normalized[np.newaxis, :], sample_rate=OUTPUT_SR, reset=True)
    timings["stage5_limiter_s"] = round(time.perf_counter() - t0, 3)
    logger.info("[%s] Stage 5 done (limiter): %.2fs", label, timings["stage5_limiter_s"])

    return limited[0], timings


def pipeline_deepfilternet(audio_48k: np.ndarray, cv: ClearVoice, mastering, limiter) -> dict:
    """Full pipeline with DeepFilterNet3 as Stage 1."""
    label = "DeepFilterNet3"
    logger.info("[%s] Loading model...", label)
    model, state, _ = init_df()
    df_sr = state.sr()
    logger.info("[%s] Model loaded (native sr=%d)", label, df_sr)

    # Stage 1: DeepFilterNet3 denoise
    tensor_48k = torch.from_numpy(audio_48k).to(torch.float32)

    # Resample to DeepFilterNet's native rate if needed
    if OUTPUT_SR != df_sr:
        tensor_in = torchaudio.functional.resample(tensor_48k, OUTPUT_SR, df_sr)
    else:
        tensor_in = tensor_48k

    t0 = time.perf_counter()
    denoised = df_enhance(model, state, tensor_in.unsqueeze(0))
    stage1_time = time.perf_counter() - t0
    denoised_1d = denoised.squeeze(0).detach().cpu()

    # Resample back to 48kHz if DeepFilterNet operates at a different rate
    if df_sr != OUTPUT_SR:
        denoised_1d = torchaudio.functional.resample(denoised_1d, df_sr, OUTPUT_SR)

    denoised_np = denoised_1d.numpy()
    logger.info("[%s] Stage 1 done (denoise): %.2fs", label, stage1_time)

    # Save Stage 1 output for comparison
    sf.write(str(OUTPUT_DIR / "pipeline_df3_stage1.wav"), denoised_np, OUTPUT_SR)

    # Stages 2-5
    final, timings = run_stages_2_to_5(denoised_np, OUTPUT_SR, cv, mastering, limiter, label)
    timings["stage1_denoise_s"] = round(stage1_time, 3)
    timings["total_s"] = round(sum(v for v in timings.values()), 3)

    out_path = OUTPUT_DIR / "pipeline_deepfilternet3_full.wav"
    sf.write(str(out_path), final, OUTPUT_SR)
    logger.info("[%s] Pipeline complete: %s (%.2fs total)", label, out_path, timings["total_s"])

    return {"label": label, "output": str(out_path), "timings": timings}


def pipeline_dpdfnet(audio_48k: np.ndarray, cv: ClearVoice, mastering, limiter) -> dict:
    """Full pipeline with DPDFNet-2 48kHz as Stage 1 (runs in .bench-venv)."""
    label = "DPDFNet-2-48k"
    logger.info("[%s] Running in isolated venv...", label)

    stage1_out = str(OUTPUT_DIR / "pipeline_dpdfnet2_stage1.wav")
    script = f"""
import time, numpy as np, soundfile as sf, dpdfnet
audio, sr = sf.read("{WAV_48K}", dtype="float32")
t0 = time.perf_counter()
enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet2_48khz_hr")
elapsed = time.perf_counter() - t0
sf.write("{stage1_out}", enhanced, sr)
print(f"{{elapsed:.3f}}")
"""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [".bench-venv/bin/python", "-c", script],
        capture_output=True, text=True, timeout=300,
    )
    wall_time = time.perf_counter() - t0

    if proc.returncode != 0:
        logger.error("[%s] FAILED: %s", label, proc.stderr[:500])
        return {"label": label, "error": proc.stderr[:500]}

    stage1_time = float(proc.stdout.strip().split("\n")[-1])
    logger.info("[%s] Stage 1 done (denoise): %.2fs (wall: %.2fs)", label, stage1_time, wall_time)

    # Load Stage 1 output
    denoised_np, sr = sf.read(stage1_out, dtype="float32")
    assert sr == OUTPUT_SR

    # Stages 2-5
    final, timings = run_stages_2_to_5(denoised_np, OUTPUT_SR, cv, mastering, limiter, label)
    timings["stage1_denoise_s"] = round(stage1_time, 3)
    timings["stage1_wall_s"] = round(wall_time, 3)
    timings["total_s"] = round(stage1_time + sum(v for k, v in timings.items() if k.startswith("stage") and k != "stage1_denoise_s" and k != "stage1_wall_s"), 3)

    out_path = OUTPUT_DIR / "pipeline_dpdfnet2_full.wav"
    sf.write(str(out_path), final, OUTPUT_SR)
    logger.info("[%s] Pipeline complete: %s (%.2fs total)", label, out_path, timings["total_s"])

    return {"label": label, "output": str(out_path), "timings": timings}


def main():
    print("=" * 70)
    print("FULL PIPELINE BENCHMARK: DeepFilterNet3 vs DPDFNet-2-48kHz")
    print("Pipeline: Denoiser → ClearVoice → Pedalboard → LUFS → Limiter")
    print(f"Input: {INPUT_M4A}")
    print("=" * 70)

    ensure_input()
    audio_48k, sr = sf.read(str(WAV_48K), dtype="float32")
    duration = len(audio_48k) / sr
    print(f"Loaded: {len(audio_48k)} samples at {sr} Hz ({duration:.1f}s)\n")

    # Shared components
    logger.info("Loading shared models...")
    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    mastering, limiter = build_mastering_chain()
    logger.info("Shared models ready.\n")

    # Run both pipelines
    df_result = pipeline_deepfilternet(audio_48k, cv, mastering, limiter)
    dpd_result = pipeline_dpdfnet(audio_48k, cv, mastering, limiter)

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    for r in [df_result, dpd_result]:
        if "error" in r:
            print(f"\n{r['label']}: FAILED — {r['error'][:200]}")
            continue
        t = r["timings"]
        print(f"\n{r['label']}:")
        print(f"  Stage 1 (denoise):     {t['stage1_denoise_s']:>6.2f}s")
        print(f"  Stage 2 (ClearVoice):  {t['stage2_clearvoice_s']:>6.2f}s")
        print(f"  Stage 3 (mastering):   {t['stage3_mastering_s']:>6.2f}s")
        print(f"  Stage 4 (LUFS):        {t['stage4_lufs_s']:>6.2f}s")
        print(f"  Stage 5 (limiter):     {t['stage5_limiter_s']:>6.2f}s")
        print(f"  TOTAL:                 {t['total_s']:>6.2f}s  (RTF {t['total_s']/duration:.4f})")
        print(f"  Output: {r['output']}")

    # Save full results
    results_path = OUTPUT_DIR / "pipeline_results.json"
    results_path.write_text(json.dumps([df_result, dpd_result], indent=2))
    print(f"\nFull results: {results_path}")
    print(f"\nA/B compare these files:")
    print(f"  1. {df_result.get('output', 'N/A')}")
    print(f"  2. {dpd_result.get('output', 'N/A')}")


if __name__ == "__main__":
    main()
