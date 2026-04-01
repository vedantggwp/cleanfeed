"""Sweep tube saturation parameters with CORRECT signal chain.

Runs stages 1-2 once, saves intermediate, then sweeps:
  MossFormer2 output → proximity → saturation → mastering EQ → LUFS → limiter

Uses a 30s clip for fast iteration.

Run: uv run python test_tube_sweep.py
"""
import types
import sys
import torchaudio

if not hasattr(torchaudio, "backend"):
    backend_module = types.ModuleType("torchaudio.backend")
    common_module = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0,
                     bits_per_sample=0, encoding=""):
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

import logging
import os
import subprocess
import tempfile
import time

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch

from engine import OUTPUT_SR, LUFS_TARGET, LIMITER_CEILING_DB, _tube_saturate
from df.enhance import init_df, enhance as df_enhance
from clearvoice import ClearVoice
from pedalboard import (
    Compressor, HighpassFilter, HighShelfFilter, Limiter,
    LowShelfFilter, PeakFilter, Pedalboard,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

INPUT_FILE = "recording.m4a"
TMP_WAV = "/tmp/tube_sweep_input.wav"
CLIP_SECONDS = 30

# Convert and clip to 30s
subprocess.run(
    ["ffmpeg", "-i", INPUT_FILE, "-ar", str(OUTPUT_SR), "-ac", "1",
     "-t", str(CLIP_SECONDS), TMP_WAV, "-y", "-loglevel", "error"],
    check=True,
)
wav, sr = torchaudio.load(TMP_WAV)
mono = wav.mean(dim=0).flatten()
print(f"Input: {mono.numel()/sr:.1f}s at {sr}Hz")
print("=" * 60)

# --- Stage 1: DeepFilterNet ---
print("Stage 1: DeepFilterNet...")
df_model, df_state, _ = init_df()
df_sr = df_state.sr()
denoised = df_enhance(df_model, df_state, mono.unsqueeze(0))
denoised_1d = denoised.squeeze(0).detach().cpu().contiguous()
print("Stage 1 done")

# --- Stage 2: MossFormer2 ---
print("Stage 2: MossFormer2...")
clearvoice = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
tmp_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_path = tmp_in.name
        sf.write(tmp_path, denoised_1d.numpy(), df_sr)
    result_dict = clearvoice(tmp_path)
finally:
    if tmp_path is not None:
        os.unlink(tmp_path)

if isinstance(result_dict, dict):
    stage2_np = list(result_dict.values())[0]
elif isinstance(result_dict, np.ndarray):
    stage2_np = result_dict
else:
    stage2_np = np.array(result_dict, dtype=np.float32)

stage2_np = np.ascontiguousarray(stage2_np.astype(np.float32))
if stage2_np.ndim == 1:
    stage2_np = stage2_np[np.newaxis, :]
print(f"Stage 2 done. Base signal ready: {stage2_np.shape}")

# --- Mastering chain (same as engine.py) ---
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
meter = pyln.Meter(OUTPUT_SR)


def finalize(audio_2d: np.ndarray) -> np.ndarray:
    """Run mastering → LUFS → limiter on (1, samples) array."""
    mastered = mastering(audio_2d, sample_rate=OUTPUT_SR, reset=True)
    mono_lufs = np.ascontiguousarray(mastered[0].astype(np.float64))
    loudness = meter.integrated_loudness(mono_lufs)
    if np.isfinite(loudness):
        normalized = pyln.normalize.loudness(mono_lufs, loudness, LUFS_TARGET).astype(np.float32)
    else:
        normalized = mono_lufs.astype(np.float32)
    return limiter(normalized[np.newaxis, :], sample_rate=OUTPUT_SR, reset=True)


# --- Baseline (no character) ---
print("\nGenerating baseline (no studio character)...")
baseline = finalize(stage2_np.copy())
torchaudio.save("tube_C0_baseline.wav", torch.from_numpy(baseline), OUTPUT_SR)
print("  tube_C0_baseline.wav — no saturation, no proximity")

# --- Parameter sweep ---
CONFIGS = [
    # (name, drive, bias, proximity_db, description)
    ("C1_gentle",   1.2, 0.10, 2.0, "very gentle — barely there"),
    ("C2_subtle",   1.5, 0.15, 2.5, "subtle warmth"),
    ("C3_current",  2.0, 0.20, 3.0, "what you heard as 'C'"),
    ("C4_warmer",   2.0, 0.30, 3.0, "more 2nd harmonic, same drive"),
    ("C5_bassier",  2.0, 0.20, 4.0, "same saturation, more low-end"),
    ("C6_rich",     2.5, 0.25, 3.5, "pushing both"),
]

print(f"\nSweeping {len(CONFIGS)} configs (correct chain: proximity → sat → master → LUFS)...")
print("=" * 60)

for name, drive, bias, prox_db, desc in CONFIGS:
    audio = stage2_np.copy()

    # Proximity effect
    proximity = Pedalboard([LowShelfFilter(cutoff_frequency_hz=150, gain_db=prox_db, q=0.7)])
    audio = proximity(audio, sample_rate=OUTPUT_SR, reset=True)

    # Tube saturation
    audio[0] = _tube_saturate(audio[0], drive=drive, bias=bias, sample_rate=OUTPUT_SR)

    # Mastering → LUFS → Limiter
    final = finalize(audio)

    outfile = f"tube_{name}.wav"
    torchaudio.save(outfile, torch.from_numpy(final), OUTPUT_SR)
    print(f"  {outfile:30s}  d={drive} b={bias} p={prox_db}dB  — {desc}")

print(f"\n{'=' * 60}")
print("All LUFS-matched. Compare against C0_baseline.")
print("The difference should be warmth/body, NOT distortion.")
print("=" * 60)
