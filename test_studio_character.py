"""A/B/C test: generate three outputs from the same recording.

Output files:
  studio_A_none.wav     — v6 baseline (no studio character)
  studio_B_softclip.wav — proximity + symmetric tanh saturation (modern clean)
  studio_C_tube.wav     — proximity + asymmetric tube saturation (vintage warm)

Run: uv run python test_studio_character.py
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
import subprocess
import time

import torch

from engine import PodcastEngine, OUTPUT_SR

logging.basicConfig(level=logging.INFO, format="%(message)s")

INPUT_FILE = "recording.m4a"
TMP_WAV = "/tmp/studio_test_input.wav"

# Convert input to 48kHz mono WAV
subprocess.run(
    ["ffmpeg", "-i", INPUT_FILE, "-ar", str(OUTPUT_SR), "-ac", "1",
     TMP_WAV, "-y", "-loglevel", "error"],
    check=True,
)
wav, sr = torchaudio.load(TMP_WAV)
mono = wav.mean(dim=0).flatten()

print(f"\nInput: {mono.numel()/sr:.1f}s at {sr}Hz")
print("=" * 60)

engine = PodcastEngine()

variants = [
    ("none", "studio_A_none.wav"),
    ("softclip", "studio_B_softclip.wav"),
    ("tube", "studio_C_tube.wav"),
]

for character, output_file in variants:
    print(f"\n{'=' * 60}")
    print(f"Generating: {output_file} (character={character})")
    print("=" * 60)

    start = time.time()
    result, out_sr = engine.enhance(mono.clone(), sr, character=character)
    elapsed = time.time() - start

    torchaudio.save(output_file, result.unsqueeze(0), out_sr)
    print(f"  -> {output_file}: {result.shape[0]/out_sr:.1f}s at {out_sr}Hz in {elapsed:.1f}s")

print(f"\n{'=' * 60}")
print("Done. Listen and compare:")
print("  A) studio_A_none.wav     — baseline (clean, no character)")
print("  B) studio_B_softclip.wav — modern warmth (odd harmonics)")
print("  C) studio_C_tube.wav     — vintage richness (even+odd harmonics)")
print("=" * 60)
