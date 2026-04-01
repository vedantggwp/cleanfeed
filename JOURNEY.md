# Project Resonance — Build Journey

> Local, privacy-first audio restoration pipeline. Phone recording in → podcast-quality audio out.

## The Goal

Record a voice memo on your phone, run one command, get podcast-quality output. No cloud. No subscriptions. No Adobe. Everything runs locally on Apple Silicon.

## Phase 1: The Architecture (2026-03-30)

Started with a clean 3-module architecture based on `resemble-enhance`, an open-source model from Resemble AI:

```
engine.py (AI model wrapper) → processor.py (OLA chunking) → cli.py (user interface)
```

**Key design decisions:**
- `engine.py` handles zero file I/O — pure tensor-in, tensor-out
- `processor.py` implements Overlap-Add with Hanning window crossfade to prevent audio clicking at chunk boundaries
- Strict MPS-first device routing for Apple Silicon, CPU fallback with warning
- `uv` for dependency management (not pip/conda)
- Signal handlers + `atexit` for clean shutdown — model unloading, MPS cache flush, temp file cleanup

**Expert review caught 4 issues before first run:**
1. Wrong import path (`resemble_enhance.enhancer` vs `resemble_enhance.enhancer.inference`) — would have crashed on import
2. MPS detection needed both `is_available()` AND `is_built()` — wrong PyTorch build would silently fall back to CPU
3. torch version floor needed (`>=2.1.0`) to guarantee MPS support
4. `lambd=0.9` was too aggressive — reviewer correctly identified it would degrade speech quality

## Phase 2: First Run — The NumPy Wall (2026-03-30)

**Problem:** `resemble-enhance` was written against NumPy 1.x. NumPy 2.0+ broke implicit scalar conversion.

```
Error: only 0-dimensional arrays can be converted to Python scalars
```

**Fix:** Pinned `numpy<2.0` (installed 1.26.4). Root cause: `resemble-enhance` hasn't been updated for NumPy 2.x breaking changes.

## Phase 3: The MPS Disaster (2026-03-30)

Ran the full pipeline. Output: **pure noise.**

Built a diagnostic script (`diagnose.py`) to isolate the problem layer by layer:

| Test | Result | Verdict |
|---|---|---|
| Denoise only (MPS) | Recognizable but degraded | Denoiser works on MPS |
| Full enhance (CPU) | Louder but robotic | CFM enhancement adds artifacts even on CPU |
| Full enhance (MPS) | Pure noise | **MPS completely breaks the CFM ODE solver** |

**Root cause:** The Continuous Flow Matching stage runs 64 sequential ODE solver steps. MPS has float32 precision issues with iterative numerical solvers — errors compound across steps until the output is pure noise. This is a known class of issue with Apple's MPS backend.

## Phase 4: The Parameter Sweep (2026-03-30)

Refused to give up on `resemble-enhance`. Ran a 7-configuration parameter sweep on CPU, varying `nfe` (solver steps), `lambd` (denoise strength), and `tau` (temperature):

| Config | nfe | lambd | tau | Time | Result |
|---|---|---|---|---|---|
| light_clean | 32 | 0.1 | 0.1 | 57s | Lightest touch |
| medium_clean | 32 | 0.5 | 0.1 | 85s | Moderate |
| heavy_denoise_low_temp | 32 | 0.9 | 0.1 | 87s | Heavy denoise |
| original_settings | 64 | 0.5 | 0.5 | 98s | Previous default |
| original_low_temp | 64 | 0.5 | 0.1 | 132s | Low randomness |
| heavy_denoise_more_steps | 64 | 0.9 | 0.1 | 189s | Max denoise |
| max_steps_low_temp | 128 | 0.5 | 0.1 | 277s | Maximum quality attempt |

**Result:** All 7 configurations had a "tearing" effect at louder passages. Noise was reduced but the output was not podcast quality. The CFM generative model was over-processing the audio — it was **re-synthesizing** the voice rather than cleaning it up.

## Phase 5: The Research Pivot (2026-03-30)

Researched how **professional podcast audio engineers** actually process audio. Key finding:

> Adobe Podcast "Enhance Speech" is ALSO a generative re-synthesizer. When it fails, it outputs English-sounding babble — proving it generates speech tokens, not just filters them.

**The professional processing chain (in order):**
1. Noise Gate/Reduction
2. High-Pass Filter (80Hz — removes rumble)
3. Subtractive EQ (cut mud at 200-400Hz)
4. Compressor (3:1 ratio, serial compression preferred)
5. De-Esser (tame sibilance at 4-10kHz)
6. Additive EQ (presence at 2-4kHz, air at 8-12kHz)
7. Loudness Normalization (-16 LUFS)
8. Brick-Wall Limiter (-1.5dB ceiling)

**Key insight:** We needed a **discriminative denoiser** (removes what shouldn't be there) followed by a **traditional DSP mastering chain** (shapes what's left). Not a generative model trying to re-imagine the audio.

## Phase 6: The Hybrid Pipeline (2026-03-30)

Rebuilt the engine with:
- **DeepFilterNet3** (1M params, real-time on CPU) for noise suppression
- **Spotify's Pedalboard** for the professional DSP mastering chain
- **pyloudnorm** for LUFS loudness normalization

First attempt used `resemble-enhance` denoiser + pedalboard. Result: still too noisy. The resemble-enhance denoiser wasn't aggressive enough.

Swapped to DeepFilterNet3. Required monkey-patching `torchaudio.backend` (removed in torchaudio 2.9+, but DeepFilterNet still imports it).

**Final pipeline:**
```
Input (.m4a/.wav/any format)
  → ffmpeg convert to 48kHz mono WAV
  → DeepFilterNet3 noise suppression (full file, real-time)
  → High-Pass Filter (80Hz)
  → Subtractive EQ (-3dB at 300Hz)
  → Dual Compressor (2:1 then 3:1, serial)
  → De-Esser (-4dB at 6kHz)
  → Presence Boost (+2.5dB at 3kHz)
  → Air Boost (+2dB at 10kHz)
  → LUFS Normalization to -16 LUFS
  → Brick-Wall Limiter (-1.5dB)
Output (.wav)
```

**Result:** 75-80% there. Voice is clean, noise is gone, compression and EQ are working. Slightly too loud — needs LUFS target tuning.

## Remaining Work

- [ ] Tune LUFS target (try -18 or -19 instead of -16)
- [ ] Integrate DeepFilterNet into `engine.py` properly (replace resemble-enhance denoiser)
- [ ] Remove the torchaudio monkey-patch (create a proper compatibility layer)
- [ ] Update `processor.py` — DeepFilterNet processes full files in real-time, OLA chunking may not be needed
- [ ] Update `app.py` Gradio UI for the new pipeline
- [ ] A/B test against Adobe Podcast Enhance on the same recording

## Tech Stack (Final)

| Component | Library | Purpose |
|---|---|---|
| Noise Suppression | DeepFilterNet3 | ML-based noise removal (1M params, real-time) |
| DSP Mastering | Spotify Pedalboard | HPF, EQ, compression, de-essing, limiting |
| Loudness | pyloudnorm | LUFS measurement and normalization |
| Audio I/O | torchaudio + ffmpeg | Format conversion and file handling |
| Package Manager | uv | Python dependency management |

## What We Learned

1. **Generative ≠ better.** For decent input audio, a discriminative denoiser + traditional DSP chain beats a generative re-synthesizer. The generative approach (resemble-enhance CFM) hallucinates artifacts on clean-ish audio.

2. **MPS is not CUDA.** Apple's MPS backend has precision issues with iterative numerical solvers (ODE/SDE). Single forward-pass models work fine; 64-step flow matching does not.

3. **The professional podcast chain is 8 specific steps in a specific order.** Order matters because audio processing is non-commutative. Compress-then-EQ ≠ EQ-then-compress.

4. **DeepFilterNet3 (1M params) outperformed resemble-enhance denoiser (10M params)** for this use case. Purpose-built tools beat general-purpose tools.

5. **The "tearing at loudness" was caused by the absence of a limiter and compressor** — the raw model output had no dynamic range control. Adding the professional mastering chain fixed it.

## Phase 7: ClearVoice MossFormer2 (2026-03-31)

**Problem:** v2 (DeepFilterNet + Pedalboard) had noise gone but lacked richness. v3 (DeepFilterNet + FlashSR + Pedalboard) introduced popping sounds from chunk boundary artifacts in FlashSR.

**Discovery:** ClearerVoice-Studio (Alibaba, 4k stars) bundles MossFormer2_SE_48K — a discriminative speech enhancement model that processes at 48kHz natively. It handles enhancement + quality improvement in one pass, no chunking needed.

**Dependency hell:** AudioSR pinned numpy==1.23.5 (won't build on Python 3.13). ClearVoice pinned librosa==0.10.2 and soundfile==0.12.1. Had to carefully resolve conflicts by removing competing pins.

**v4 test (MossFormer2 only):** Quality improved but noise came back — MossFormer2 enhances but doesn't aggressively denoise. Needed both models.

**v5 final pipeline (the breakthrough):**
```
DeepFilterNet3 (noise kill) → MossFormer2 (enhance) → Pedalboard (master) → LUFS (-18) → Limiter
```

**Result:** ~85% podcast quality. Noise eliminated, loudness correct, no artifacts, no popping. Entire 2-minute recording processes in ~15 seconds total (DeepFilterNet ~3s + MossFormer2 ~7s + DSP instant).

**Remaining gap:** Studio mic character — proximity effect warmth (80-250Hz), harmonic saturation/richness, voice-specific EQ tuning. This is the "last mile" problem — the difference between clean audio and audio that sounds like it was recorded on a $500 condenser mic.

## Phase 8: The Last Mile — Integration & Studio Character Experiments (2026-04-01)

### Code Integration (v6)

Integrated the v5 pipeline (previously a standalone test script) into the proper codebase:
- **engine.py** — replaced FlashSR with MossFormer2_SE_48K via ClearVoice. LUFS target adjusted from -16 to -18. Uses ClearVoice file I/O mode (not tensor-to-tensor) because t2t mode OOMs on MPS for >60s audio — file I/O mode handles internal 4s sliding-window segmentation.
- **processor.py** — removed OLA chunking entirely (115 → 55 lines). Both DeepFilterNet and MossFormer2 handle their own segmentation, so external chunking was redundant and caused artifacts.
- **cli.py** — ffmpeg conversion target changed from 16kHz to 48kHz to match the pipeline's native sample rate.

### FINALLY Research (Dead End)

Investigated Samsung's FINALLY (NeurIPS 2024) via the inverse-ai/FINALLY-Speech-Enhancement repo:
- **No pretrained weights available.** Training from scratch requires multi-GPU, LibriTTS-R + DAPS-clean datasets, and a 3-stage pipeline with known NaN stability issues.
- **Voice identity risk.** Generative GAN approach can shift accents and change speaker voice in low-SNR regions. Dealbreaker for podcast use.
- **MOS > ground truth (4.63 vs 4.56)** — model adds aesthetic coloration, not faithful restoration. Same fundamental problem as resemble-enhance CFM.
- **One interesting idea:** learnable 16→48kHz upsampling (Upsample WaveUNet). Could be explored as a standalone module later.

### Studio Mic Character Experiments (Ruled Out)

Tested two DSP approaches to close the "studio mic character" gap:

**A) Proximity effect + soft-clip saturation (tanh):** Symmetric waveshaping, odd harmonics only. Result: voice sounds "sleepy/meditative" — rounds off transients, removes speech energy. Good for ambient content, wrong for podcasts.

**B) Proximity effect + tube simulation (asymmetric waveshaping):** Even + odd harmonics, models vacuum tube nonlinearity. Result: adds warmth/body at subtle settings, but even conservative parameters (drive=1.5, bias=0.15) sounded like added coloration rather than natural quality improvement. At higher settings, sounds like a cheap old wired mic.

**Key finding: MossFormer2 already does its own spectral enhancement.** Layering proximity + saturation on top of an already-enhanced signal adds coloration to coloration. The clean v6 pipeline (no studio character) consistently sounded best in A/B testing. The "last 15%" gap may be smaller than estimated, or it requires a fundamentally different approach (learned upsampling, voice-specific fine-tuning) rather than DSP post-processing.

**v6 is the final pipeline:**
```
Input (.m4a/.wav/any) → ffmpeg 48kHz mono → DeepFilterNet3 (denoise) → MossFormer2 (enhance) → Pedalboard DSP (HPF/EQ/compress/de-ess/presence/air) → LUFS -18 → Limiter -1.5dB → Output
```

Processing time: ~14s for 2 minutes of audio on Apple M4.

## Version History

| Version | Pipeline | Result |
|---------|----------|--------|
| v1 | resemble-enhance (full CFM, MPS) | Pure noise |
| v1b | resemble-enhance (full CFM, CPU) | Robotic, artifacts |
| sweep | 7 parameter configs on CPU | All had tearing |
| v2 | resemble-enhance denoise + Pedalboard | 75-80%, noise still present |
| v3 | DeepFilterNet + FlashSR + Pedalboard | Popping at chunk boundaries |
| v4 | MossFormer2 + Pedalboard | Good but noise returned |
| v5 | DeepFilterNet + MossFormer2 + Pedalboard | ~85%, noise gone, standalone script |
| v6 | v5 integrated into engine.py + processor.py | Production-ready, ~85-90% |
| v6+sat | v6 + proximity + saturation (softclip/tube) | Ruled out — coloration, not improvement |

## What We Learned (Updated)

1. **Generative ≠ better.** For decent input audio, discriminative models beat generative re-synthesizers.

2. **MPS is not CUDA.** Iterative ODE solvers produce noise on Apple MPS. Single-pass models (DeepFilterNet, MossFormer2) work fine.

3. **Order matters.** The professional podcast chain is 8 steps in a specific order. Signal chain is non-commutative: saturation→compression ≠ compression→saturation.

4. **Purpose-built > general-purpose.** DeepFilterNet (1M params) outperformed resemble-enhance denoiser (10M params).

5. **No limiter = tearing.** Raw model output needs dynamic range control.

6. **Chunking creates artifacts.** Processing full files in one pass is better than OLA chunking when the model supports it.

7. **Two specialized models > one generalist.** DeepFilterNet (noise only) + MossFormer2 (enhance only) outperformed either alone.

8. **Dependency pinning is the real enemy.** More time fighting dependency conflicts than writing code.

9. **Don't layer enhancement on enhancement.** MossFormer2 already does spectral shaping. Adding DSP saturation/proximity on top of it adds coloration, not quality. The clean pipeline is the best pipeline.

10. **Generative models are a voice identity risk.** FINALLY (Samsung, NeurIPS 2024) achieves MOS scores above ground truth but can shift accents and change speaker voice. For content where voice identity matters, discriminative models are safer.

11. **ClearVoice tensor-to-tensor mode OOMs on long audio.** File I/O mode handles internal segmentation (4s sliding windows). Always use file I/O for production; t2t only for short clips.
