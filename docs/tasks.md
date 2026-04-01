# Execution Roadmap

**STATUS:** Phases 1-8 COMPLETE. Pipeline production-ready.

## Phase 1-3: Foundation (COMPLETE)
Engine, chunking pipeline, and Gradio UI built on resemble-enhance.

## Phase 4-5: Research Pivot (COMPLETE)
Discovered generative models (resemble-enhance CFM) are wrong tool for this job.
Pivoted to discriminative denoiser + professional DSP chain.

## Phase 6: Hybrid Pipeline (COMPLETE)
DeepFilterNet3 + Pedalboard mastering chain. 75-80% quality.

## Phase 7: MossFormer2 (COMPLETE)
Added ClearVoice MossFormer2_SE_48K as Stage 2. Achieved ~85% quality.

## Phase 8: Integration & Experiments (COMPLETE)
- Integrated v5 pipeline into engine.py/processor.py (v6)
- Researched FINALLY (Samsung NeurIPS 2024) — dead end (no weights, voice identity risk)
- Tested proximity effect + saturation (softclip/tube) — ruled out (coloration, not improvement)
- Clean v6 pipeline confirmed as best output

## Future Directions (Not Planned)
- Voice-specific fine-tuning (train a model on paired phone/studio recordings)
- Learned 16→48kHz upsampling (FINALLY's Upsample WaveUNet idea, standalone)
- Open-source release preparation
