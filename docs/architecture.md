# System Architecture: Project Resonance

## Tech Stack
* **Python Manager:** `uv`
* **Noise Suppression:** DeepFilterNet3 (1M params, real-time on CPU)
* **Speech Enhancement:** MossFormer2_SE_48K via ClearerVoice-Studio (Alibaba)
* **DSP Mastering:** Spotify Pedalboard (HPF, EQ, compression, de-essing, limiting)
* **Loudness:** pyloudnorm (LUFS measurement and normalization)
* **Audio I/O:** torchaudio + soundfile + ffmpeg
* **UI:** Gradio

## Pipeline (5 stages)

```
Input (.m4a/.wav/any format)
  → ffmpeg convert to 48kHz mono WAV
  → Stage 1: DeepFilterNet3 noise suppression
  → Stage 2: MossFormer2_SE_48K speech enhancement
  → Stage 3: Pedalboard DSP mastering chain
      → High-Pass Filter (80Hz)
      → Subtractive EQ (-3dB at 300Hz)
      → Dual Compressor (2:1 then 3:1)
      → De-Esser (-4dB at 6kHz)
      → Presence Boost (+2.5dB at 3kHz)
      → Air Boost (+2dB at 10kHz)
  → Stage 4: LUFS normalization to -18 LUFS
  → Stage 5: Brick-wall limiter (-1.5dB)
Output (.wav, 48kHz mono)
```

## Module Definitions (Strict Separation)

### 1. `engine.py` (The AI + DSP Core)
* **Purpose:** Wraps both ML models and the DSP mastering chain.
* **Input:** A 1D mono `torch.Tensor` and sample rate.
* **Process:** Runs the 5-stage pipeline. Uses ClearVoice file I/O mode for MossFormer2 (internal segmentation prevents MPS OOM on long audio).
* **Output:** A podcast-quality `torch.Tensor` at 48kHz.
* **Constraint:** No user-facing file I/O. Internal temp files for MossFormer2 are created and cleaned up within the method.

### 2. `processor.py` (The Audio Loader)
* **Purpose:** Loads audio, converts to mono, passes to engine, saves output.
* **Input:** File path to any supported audio format.
* **Process:** Loads via torchaudio, averages channels to mono, calls `engine.enhance()`, saves result.
* **Output:** Enhanced `.wav` file on disk.
* **Note:** OLA chunking was removed in v6 — both models handle segmentation internally.

### 3. `cli.py` (The CLI)
* **Purpose:** Command-line interface with format conversion and cleanup.
* **Process:** Validates input, converts non-WAV via ffmpeg to 48kHz mono, calls `processor.process_audio()`, reports duration and sample rate.

### 4. `app.py` (The Web UI)
* **Purpose:** Drag-and-drop Gradio interface with A/B comparison player.
* **Process:** Accepts upload, passes to processor, displays original vs enhanced audio players.
