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

## Phase 8: The Last Mile (NEXT)

Candidates for closing the remaining 15%:
- **Proximity effect simulation** — low-shelf warmth boost tuned for voice
- **Harmonic saturation** — subtle tube/tape saturation (pedalboard has no saturation, may need a VST plugin loaded via pedalboard's VST3 loader)
- **FINALLY (Samsung, NeurIPS 2024)** — unofficial implementation exists, claims "studio-like quality" from any input
- **Voice-specific EQ** — analyze the recording's spectral profile and tune EQ to complement rather than generic settings
- **Stereo widening / room simulation** — subtle early reflections that add "space" without muddiness

## Version History

| Version | Pipeline | Result |
|---------|----------|--------|
| v1 | resemble-enhance (full CFM, MPS) | Pure noise |
| v1b | resemble-enhance (full CFM, CPU) | Robotic, artifacts |
| sweep | 7 parameter configs on CPU | All had tearing |
| v2 | resemble-enhance denoise + Pedalboard | 75-80%, noise still present |
| v3 | DeepFilterNet + FlashSR + Pedalboard | Popping at chunk boundaries |
| v4 | MossFormer2 + Pedalboard | Good but noise returned |
| v5 | DeepFilterNet + MossFormer2 + Pedalboard | ~85%, noise gone, lacking studio character |

## What We Learned (Updated)

1. **Generative ≠ better.** For decent input audio, discriminative models beat generative re-synthesizers.

2. **MPS is not CUDA.** Iterative ODE solvers produce noise on Apple MPS.

3. **Order matters.** The professional podcast chain is 8 steps in a specific order.

4. **Purpose-built > general-purpose.** DeepFilterNet (1M params) outperformed resemble-enhance denoiser (10M params).

5. **No limiter = tearing.** Raw model output needs dynamic range control.

6. **Chunking creates artifacts.** Processing the full file in one pass (MossFormer2, DeepFilterNet) is better than OLA chunking when the model supports it. FlashSR's chunk-by-chunk super-resolution caused popping.

7. **Two specialized models > one generalist.** DeepFilterNet (noise only) + MossFormer2 (enhance only) outperformed MossFormer2 alone. Each model does its one job well.

8. **Dependency pinning is the real enemy.** AudioSR, resemble-enhance, and ClearVoice all pin stale numpy/librosa versions. More time was spent fighting dependency conflicts than writing code.

9. **The last 15% is mic physics, not algorithms.** The gap between "clean" and "studio quality" is proximity effect, harmonic richness, and transient detail — things a phone mic physically cannot capture. Closing this gap requires either generative modeling (FINALLY) or DSP simulation (saturation, room modeling).
