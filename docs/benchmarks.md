# Benchmark Results & Decision Record

> Date: 2026-04-01 | Input: recording.m4a (115.9s voice memo, 48kHz mono)
> Machine: Apple Silicon M4, macOS

## Decision 1: Denoiser — DeepFilterNet3 (KEEP)

### Candidates

| Denoiser | Native SR | Speed (Stage 1) | RTF | Full Pipeline | API |
|----------|-----------|-----------------|-----|---------------|-----|
| **DeepFilterNet3** | 48kHz | 2.86s | 0.025 | 6.90s | PyTorch tensor |
| DPDFNet-2 48kHz HR | 48kHz | 13.82s | 0.119 | 17.25s | numpy via ONNX |
| DPDFNet-4 (16kHz) | 16kHz | 35.42s | 0.306 | — | numpy via ONNX |

### Listening Test (full pipeline output)

- **DeepFilterNet3 → ClearVoice → Pedalboard**: Preferred. More enjoyable, slightly louder perceived presence.
- **DPDFNet-2 → ClearVoice → Pedalboard**: Good but not better. Close call. Slight noise coupled to voice at high volume (causal model artifact — noise gate activates only during speech energy).
- DPDFNet in isolation (no ClearVoice) sounded clearer than ClearVoice alone, but once both pass through the full pipeline, DeepFilterNet3 wins.

### Decision Rationale

1. **Quality**: DeepFilterNet3 preferred in blind A/B (full pipeline)
2. **Speed**: 2.5x faster total pipeline (6.9s vs 17.3s)
3. **Dependencies**: DPDFNet requires `librosa>=0.11`, conflicts with ClearVoice's `librosa==0.10.2.post1`
4. **Maturity**: DeepFilterNet3 is battle-tested in this pipeline since Phase 6

### DPDFNet Notes (for future reference)

- `pip install dpdfnet` — clean API: `dpdfnet.enhance(audio, sample_rate, model)`
- Has `StreamEnhancer` for real-time use (~20ms latency)
- ONNX-only (CPU), no MPS/CUDA acceleration via pip
- Worth revisiting for Sprint 5 (real-time on-device) if they ship GPU-accelerated inference
- GitHub: ceva-ip/DPDFNet | PyPI: dpdfnet

---

## Decision 2: ClearVoice Mode — Numpy (SWITCH)

### Benchmark

| Mode | Time | RTF | OOM | Temp Files |
|------|------|-----|-----|------------|
| File I/O (current) | 10.93s | 0.094 | None | Yes (write + read) |
| **Numpy (new)** | 3.44s | 0.030 | None | None |

### Output Comparison

- Outputs are NOT bit-exact (max diff 0.1158, mean diff 0.0003)
- Numpy mode slightly bassier in isolated listening
- Once through full pipeline, difference is inaudible
- Length matches exactly (5,561,344 samples)

### Decision Rationale

1. **3.2x faster** — eliminates file write/read overhead
2. **No OOM** on 116s clip with `torch.no_grad()` wrapper
3. **Cleaner architecture** — eliminates temp file I/O from engine.py, closer to "tensor-in, tensor-out" ideal
4. Internal segmentation (4s overlapping windows) works identically in numpy mode

### Implementation Notes

- Pass `[1, length]` shaped float32 numpy array
- Wrap call in `torch.no_grad()` — ClearVoice doesn't do this internally (GitHub issue #131)
- MPS is disabled in ClearVoice's numpy path — falls back to CPU automatically
- Single model only in numpy mode (we only use MossFormer2_SE_48K, so fine)

---

## Pipeline Timing Breakdown (final)

```
DeepFilterNet3 pipeline (WINNER):
  Stage 1 (DeepFilterNet3):  2.86s  ████░░░░░░░░░░░░░░░░  41%
  Stage 2 (ClearVoice):      3.79s  █████░░░░░░░░░░░░░░░  55%
  Stage 3 (Pedalboard):      0.12s  ░░░░░░░░░░░░░░░░░░░░   2%
  Stage 4 (LUFS):            0.08s  ░░░░░░░░░░░░░░░░░░░░   1%
  Stage 5 (Limiter):         0.04s  ░░░░░░░░░░░░░░░░░░░░   1%
  TOTAL:                     6.90s  RTF 0.060

  With numpy ClearVoice (projected):
  Stage 2 drops from ~10.9s → ~3.4s
  Estimated new total: ~6.5s  RTF ~0.056
```

---

## Files

| File | Purpose |
|------|---------|
| `benchmark_outputs/pipeline_deepfilternet3_full.wav` | Full pipeline with DeepFilterNet3 (WINNER) |
| `benchmark_outputs/pipeline_dpdfnet2_full.wav` | Full pipeline with DPDFNet-2 48kHz |
| `benchmark_outputs/pipeline_df3_stage1.wav` | DeepFilterNet3 denoise only |
| `benchmark_outputs/pipeline_dpdfnet2_stage1.wav` | DPDFNet-2 denoise only |
| `benchmark_outputs/clearvoice_file_io.wav` | ClearVoice file I/O mode |
| `benchmark_outputs/clearvoice_numpy.wav` | ClearVoice numpy mode |
| `benchmark_outputs/results.json` | Denoiser metrics |
| `benchmark_outputs/pipeline_results.json` | Full pipeline metrics |
