# Manifest

## Key Files
- `engine.py` — 5-stage pipeline: DeepFilterNet → MossFormer2 → Pedalboard → LUFS → Limiter
- `processor.py` — Audio loader, mono conversion, engine passthrough, file save
- `cli.py` — CLI interface with ffmpeg format conversion
- `app.py` — Gradio web UI with A/B comparison player
- `JOURNEY.md` — Full build log from idea to working pipeline (Phases 1-8)
- `CLAUDE.md` — Agent identity, architecture rules, execution protocol
- `docs/architecture.md` — System architecture and module definitions
- `docs/tasks.md` — Execution roadmap and phase status
- `docs/references.md` — 30+ models/repos/papers evaluated
- `docs/knowledge-base.md` — Model evaluations, API reference, parameter guide
- `setup.sh` — First-time setup script (uv dependencies)
- `recording.m4a` — Test input (voice memo)
- `podcast_v6_final.wav` — Current best output

## Test Scripts
- `test_full_pipeline.py` — Original v2 pipeline test (DeepFilterNet + Pedalboard, no MossFormer2)
- `test_deepfilter.py` — DeepFilterNet isolation test
- `test_studio_character.py` — A/B/C test: none vs softclip vs tube saturation
- `test_tube_sweep.py` — Parameter sweep for tube saturation (6 configs)
- `test_processor.py` — Original processor test (sine wave)
- `diagnose.py` — Layer-by-layer diagnostic (MPS vs CPU, denoise vs enhance)
- `sweep.py` — resemble-enhance 7-config parameter sweep

## Recent Changes
- 2026-04-01: Rewrote `engine.py` — FlashSR replaced with MossFormer2, LUFS -18, clean 5-stage pipeline
- 2026-04-01: Simplified `processor.py` — removed OLA chunking (115 → 55 lines)
- 2026-04-01: Fixed `cli.py` — ffmpeg converts to 48kHz (was 16kHz)
- 2026-04-01: Updated `docs/architecture.md` — reflects v6 pipeline
- 2026-04-01: Updated `docs/tasks.md` — Phase 8 complete
- 2026-04-01: Updated `JOURNEY.md` — Phase 8 with FINALLY research and saturation experiments
- 2026-04-01: Created `MANIFEST.md`
