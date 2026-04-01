"""Benchmark: ClearVoice file I/O mode vs Numpy2Numpy mode.

Tests if the numpy API eliminates temp file I/O without OOM,
and compares speed and output quality.
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

import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from clearvoice import ClearVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_WAV = Path("benchmark_outputs/input_48k.wav")
OUTPUT_DIR = Path("benchmark_outputs")


def test_file_io_mode(cv: ClearVoice, audio: np.ndarray, sr: int) -> dict:
    """Current approach: write temp file, pass path to ClearVoice."""
    logger.info("Testing FILE I/O mode...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, sr)

        t0 = time.perf_counter()
        result = cv(tmp_path)
        elapsed = time.perf_counter() - t0
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    if isinstance(result, dict):
        output = list(result.values())[0]
    elif isinstance(result, np.ndarray):
        output = result
    else:
        output = np.array(result, dtype=np.float32)

    output = output.flatten().astype(np.float32)
    out_path = OUTPUT_DIR / "clearvoice_file_io.wav"
    sf.write(str(out_path), output, sr)

    rms = float(np.sqrt(np.mean(output**2)))
    peak = float(np.max(np.abs(output)))

    return {
        "mode": "file_io",
        "elapsed_s": round(elapsed, 3),
        "rtf": round(elapsed / (len(output) / sr), 4),
        "samples": len(output),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "output_file": str(out_path),
    }


def test_numpy_mode(cv: ClearVoice, audio: np.ndarray, sr: int) -> dict:
    """New approach: pass numpy array directly, no temp files."""
    logger.info("Testing NUMPY mode (with torch.no_grad)...")

    # Ensure correct shape: [1, length] for mono
    if audio.ndim == 1:
        input_array = audio[np.newaxis, :]
    else:
        input_array = audio

    input_array = input_array.astype(np.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        result = cv(input_array)
    elapsed = time.perf_counter() - t0

    if isinstance(result, dict):
        output = list(result.values())[0]
    elif isinstance(result, np.ndarray):
        output = result
    elif isinstance(result, torch.Tensor):
        output = result.numpy()
    else:
        output = np.array(result, dtype=np.float32)

    output = output.flatten().astype(np.float32)
    out_path = OUTPUT_DIR / "clearvoice_numpy.wav"
    sf.write(str(out_path), output, sr)

    rms = float(np.sqrt(np.mean(output**2)))
    peak = float(np.max(np.abs(output)))

    return {
        "mode": "numpy",
        "elapsed_s": round(elapsed, 3),
        "rtf": round(elapsed / (len(output) / sr), 4),
        "samples": len(output),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "output_file": str(out_path),
    }


def main():
    print("=" * 60)
    print("CLEARVOICE BENCHMARK: File I/O vs Numpy2Numpy")
    print(f"Input: {INPUT_WAV}")
    print("=" * 60)

    if not INPUT_WAV.exists():
        print(f"ERROR: {INPUT_WAV} not found. Run benchmark_denoisers.py first.")
        return

    audio, sr = sf.read(str(INPUT_WAV), dtype="float32")
    print(f"Loaded: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.1f}s)")

    logger.info("Loading MossFormer2_SE_48K...")
    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    logger.info("Model loaded.")

    # Run both modes
    file_result = test_file_io_mode(cv, audio, sr)
    numpy_result = test_numpy_mode(cv, audio, sr)

    # Compare
    print(f"\n{'=' * 60}")
    print(f"{'Mode':<15} {'Time (s)':>10} {'RTF':>8} {'RMS':>10} {'Peak':>8} {'Samples':>10}")
    print("-" * 63)
    for r in [file_result, numpy_result]:
        print(f"{r['mode']:<15} {r['elapsed_s']:>10.2f} {r['rtf']:>8.4f} {r['rms']:>10.6f} {r['peak']:>8.4f} {r['samples']:>10}")

    # Bit-exact comparison
    file_audio, _ = sf.read(file_result["output_file"], dtype="float32")
    numpy_audio, _ = sf.read(numpy_result["output_file"], dtype="float32")

    min_len = min(len(file_audio), len(numpy_audio))
    if min_len > 0:
        max_diff = float(np.max(np.abs(file_audio[:min_len] - numpy_audio[:min_len])))
        mean_diff = float(np.mean(np.abs(file_audio[:min_len] - numpy_audio[:min_len])))
        print(f"\nOutput comparison (file vs numpy):")
        print(f"  Max diff:  {max_diff:.8f}")
        print(f"  Mean diff: {mean_diff:.8f}")
        print(f"  Bit-exact: {'YES' if max_diff == 0 else 'NO'}")
        print(f"  Length match: {len(file_audio)} vs {len(numpy_audio)} ({'YES' if len(file_audio) == len(numpy_audio) else 'NO'})")

    print(f"\nConclusion:")
    speedup = file_result["elapsed_s"] / numpy_result["elapsed_s"] if numpy_result["elapsed_s"] > 0 else 0
    if speedup > 1:
        print(f"  Numpy mode is {speedup:.1f}x FASTER")
    else:
        print(f"  File I/O mode is {1/speedup:.1f}x FASTER")

    if max_diff < 0.001:
        print(f"  Outputs are effectively identical (max diff {max_diff:.8f})")
    else:
        print(f"  Outputs DIFFER significantly (max diff {max_diff:.4f}) — listen to both!")


if __name__ == "__main__":
    main()
