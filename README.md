# cleanfeed

Local AI audio restoration. Phone recording → podcast quality.

**Zero cloud. Zero uploads. Everything runs on your machine.**

cleanfeed is a 5-stage audio enhancement pipeline that transforms noisy voice memos into broadcast-ready audio. It combines neural noise suppression, speech enhancement, and professional mastering — all running locally on CPU.

## Before / After

> Audio demos coming soon — record on your phone, run `cleanfeed`, hear the difference.

<!-- TODO: Add audio player embeds once demo files are hosted -->

## Install

```bash
pip install cleanfeed
```

Requires Python 3.11+ and ffmpeg (`brew install ffmpeg` on macOS).

## Usage

### CLI (simplest)

```bash
cleanfeed recording.m4a podcast.wav
```

### Python API

```python
import cleanfeed

# One-liner: file in, file out
cleanfeed.enhance("recording.m4a", "podcast.wav")

# Advanced: tensor-level control
engine = cleanfeed.Engine()
enhanced_tensor, sample_rate = engine.enhance(audio_tensor, input_sr)
```

### Web UI

```bash
pip install cleanfeed[ui]
python -m cleanfeed.app
# Opens at http://localhost:7860
```

## What it does

Five stages, each doing one thing well:

| Stage | Model / Tool | What it does |
|-------|-------------|-------------|
| 1 | DeepFilterNet3 | Neural noise suppression — removes background noise |
| 2 | MossFormer2 (48kHz) | Speech enhancement — clarifies voice detail |
| 3 | Pedalboard | DSP mastering — EQ, compression, de-essing, air boost |
| 4 | pyloudnorm | Loudness normalization to -18 LUFS (podcast standard) |
| 5 | Brick-wall limiter | Prevents clipping at -1.5 dB ceiling |

Processing a 2-minute recording takes ~7 seconds on Apple Silicon.

## How it started

cleanfeed began as a personal problem: voice memos recorded on a phone sound terrible in a podcast. The AI models that exist are research demos, not products. Professional mastering chains exist but don't denoise. Nothing combines both into a single, local pipeline.

So I built it. The full build story — from first prototype to production pipeline, every dead end and breakthrough — is in [JOURNEY.md](JOURNEY.md).

## Architecture

```
Input (any format)
  → ffmpeg → 48kHz mono WAV
  → Stage 1: DeepFilterNet3 (noise suppression)
  → Stage 2: MossFormer2_SE_48K (speech enhancement)
  → Stage 3: Pedalboard (HPF → EQ → compression → de-ess → presence → air)
  → Stage 4: LUFS normalization (-18 LUFS)
  → Stage 5: Brick-wall limiter (-1.5 dB)
Output: podcast-quality 48kHz WAV
```

Hard boundaries: the engine never touches the filesystem. The processor never touches the model. The CLI never touches tensors.

## Development

```bash
# Clone and setup
git clone https://github.com/vedantggwp/cleanfeed.git
cd cleanfeed
uv sync

# Run tests (fast unit tests only)
uv run pytest -m "not slow"

# Run full test suite (loads ML models, ~30s)
uv run pytest

# Run on a file
uv run cleanfeed recording.m4a output.wav
```

## License

MIT
