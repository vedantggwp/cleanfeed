"""Benchmark: DPDFNet vs DeepFilterNet3 on recording.m4a.

Outputs WAV files for A/B comparison + timing + basic quality metrics.
Run with main venv for DeepFilterNet, isolated venv for DPDFNet.
"""

import json
import subprocess
import sys
import time
import types
from pathlib import Path

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

import numpy as np
import soundfile as sf

INPUT_FILE = Path("recording.m4a")
OUTPUT_DIR = Path("benchmark_outputs")
WAV_48K = OUTPUT_DIR / "input_48k.wav"
WAV_16K = OUTPUT_DIR / "input_16k.wav"
RESULTS_FILE = OUTPUT_DIR / "results.json"


def convert_input():
    """Convert m4a to WAV at both 48kHz and 16kHz for the denoisers."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(INPUT_FILE), "-ar", "48000", "-ac", "1", str(WAV_48K)],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(INPUT_FILE), "-ar", "16000", "-ac", "1", str(WAV_16K)],
        capture_output=True,
        check=True,
    )
    print(f"Converted: {WAV_48K} and {WAV_16K}")


def compute_metrics(audio: np.ndarray, sr: int) -> dict:
    """Basic signal quality metrics (no reference needed)."""
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    crest_factor = float(peak / rms) if rms > 0 else 0.0
    duration = len(audio) / sr

    return {
        "duration_s": round(duration, 2),
        "sample_rate": sr,
        "samples": len(audio),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "crest_factor_db": round(20 * np.log10(crest_factor) if crest_factor > 0 else -100, 2),
    }


def benchmark_deepfilternet():
    """Run DeepFilterNet3 via the current main venv."""
    print("\n--- DeepFilterNet3 ---")
    from df.enhance import enhance as df_enhance
    from df.enhance import init_df

    model, state, _ = init_df()
    df_sr = state.sr()

    audio_16k, sr = sf.read(str(WAV_16K), dtype="float32")
    assert sr == 16000

    import torch

    tensor = torch.from_numpy(audio_16k).unsqueeze(0)

    t0 = time.perf_counter()
    enhanced = df_enhance(model, state, tensor)
    elapsed = time.perf_counter() - t0

    result = enhanced.squeeze(0).numpy()
    out_path = OUTPUT_DIR / "deepfilternet3_16k.wav"
    sf.write(str(out_path), result, df_sr)

    metrics = compute_metrics(result, df_sr)
    metrics["elapsed_s"] = round(elapsed, 3)
    metrics["rtf"] = round(elapsed / metrics["duration_s"], 4)
    print(f"  Time: {elapsed:.2f}s | RTF: {metrics['rtf']:.4f} | Output: {out_path}")
    return {"deepfilternet3_16k": metrics}


def benchmark_dpdfnet():
    """Run DPDFNet variants via the isolated .bench-venv."""
    results = {}

    models_to_test = [
        ("dpdfnet4", str(WAV_16K), 16000),
        ("dpdfnet2_48khz_hr", str(WAV_48K), 48000),
    ]

    for model_name, wav_path, expected_sr in models_to_test:
        print(f"\n--- DPDFNet: {model_name} ---")

        script = f"""
import time, json, sys
import numpy as np
import soundfile as sf
import dpdfnet

audio, sr = sf.read("{wav_path}", dtype="float32")
assert sr == {expected_sr}, f"Expected {expected_sr}, got {{sr}}"

t0 = time.perf_counter()
enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="{model_name}")
elapsed = time.perf_counter() - t0

out_path = "benchmark_outputs/{model_name}.wav"
sf.write(out_path, enhanced, sr)

rms = float(np.sqrt(np.mean(enhanced**2)))
peak = float(np.max(np.abs(enhanced)))
cf = peak / rms if rms > 0 else 0.0

result = {{
    "duration_s": round(len(enhanced) / sr, 2),
    "sample_rate": sr,
    "samples": len(enhanced),
    "rms": round(rms, 6),
    "peak": round(peak, 6),
    "crest_factor_db": round(20 * np.log10(cf) if cf > 0 else -100, 2),
    "elapsed_s": round(elapsed, 3),
    "rtf": round(elapsed / (len(enhanced) / sr), 4),
}}
print(json.dumps(result))
"""
        proc = subprocess.run(
            [".bench-venv/bin/python", "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr[:500]}")
            results[model_name] = {"error": proc.stderr[:500]}
            continue

        stdout_lines = proc.stdout.strip().split("\n")
        metrics = json.loads(stdout_lines[-1])
        results[model_name] = metrics
        print(f"  Time: {metrics['elapsed_s']:.2f}s | RTF: {metrics['rtf']:.4f} | Output: benchmark_outputs/{model_name}.wav")

    return results


def main():
    print("=" * 60)
    print("DENOISER BENCHMARK: DPDFNet vs DeepFilterNet3")
    print(f"Input: {INPUT_FILE}")
    print("=" * 60)

    convert_input()

    # Input metrics for reference
    audio_48k, _ = sf.read(str(WAV_48K), dtype="float32")
    audio_16k, _ = sf.read(str(WAV_16K), dtype="float32")
    input_metrics = {
        "input_48k": compute_metrics(audio_48k, 48000),
        "input_16k": compute_metrics(audio_16k, 16000),
    }

    all_results = {"input": input_metrics}

    # Run benchmarks
    all_results.update(benchmark_deepfilternet())
    all_results.update(benchmark_dpdfnet())

    # Save results
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'=' * 60}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"WAV files in {OUTPUT_DIR}/ — open in any player for A/B comparison")
    print(f"{'=' * 60}")

    # Summary table
    print(f"\n{'Model':<25} {'Time (s)':>10} {'RTF':>8} {'RMS':>10} {'Peak':>8} {'Crest (dB)':>12}")
    print("-" * 75)
    for name, m in all_results.items():
        if "error" in m:
            print(f"{name:<25} {'ERROR':>10}")
        elif "elapsed_s" in m:
            print(f"{name:<25} {m['elapsed_s']:>10.2f} {m['rtf']:>8.4f} {m['rms']:>10.6f} {m['peak']:>8.4f} {m['crest_factor_db']:>12.2f}")
        else:
            print(f"{name:<25} {'(input)':>10} {'':>8} {m['rms']:>10.6f} {m['peak']:>8.4f} {m['crest_factor_db']:>12.2f}")


if __name__ == "__main__":
    main()
