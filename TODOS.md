# Cleanfeed — Task List

> Shipping as `cleanfeed` on PyPI. Local, privacy-first audio restoration.

## Sprint 1: Benchmark + Validate

- [ ] **1.1** Benchmark DPDFNet vs DeepFilterNet3 on recording.m4a — compare quality, speed, API
- [ ] **1.2** Test ClearVoice Numpy2Numpy API — eliminate temp file I/O if no OOM
- [ ] **1.3** Write `docs/benchmarks.md` — decision record for denoiser choice
- [ ] **1.4** Update engine.py with winning stack, verify no regressions

## Sprint 2: Package as cleanfeed

- [ ] **2.1** Restructure flat files → `cleanfeed/` package (engine, processor, cli, app, _compat, presets)
- [ ] **2.2** Public API: `cleanfeed.enhance()`, `cleanfeed.Engine()`, `__init__.py`
- [ ] **2.3** pyproject.toml — name, entry points, classifiers, dependency cleanup
- [ ] **2.4** Lower Python floor to 3.11 — test compatibility
- [ ] **2.5** pytest suite — unit (compat, presets), integration (engine, processor), E2E (CLI, public API)
- [ ] **2.6** GitHub repo `ved-labs/cleanfeed` + README with before/after audio demos
- [ ] **2.7** First PyPI release: `pip install cleanfeed` v0.1.0

## Sprint 3: HuggingFace Space + Feedback Loop

- [ ] **3.1** Deploy Gradio app to HuggingFace Spaces (`vedant/cleanfeed`)
- [ ] **3.2** Feedback widget — star rating, issue tags, 3 consent tiers, HuggingFaceDatasetSaver
- [ ] **3.3** Quality metrics — DNSMOS/SpeechScore, spectral analysis, SNR estimation
- [ ] **3.4** CLI opt-in feedback — voluntary post-processing prompt, offline fallback

## Sprint 4: Semantic Controls (after feedback data)

- [ ] **4.1** Cluster feedback submissions by spectral profile
- [ ] **4.2** Derive presets from real-world clusters (not guesswork)
- [ ] **4.3** Build semantic control layer — Clarity, Warmth, Brightness, Loudness, Smoothness
- [ ] **4.4** AI profile suggestion from input spectral analysis
- [ ] **4.5** Consumer-facing UI with semantic controls

## Sprint 5+: Real-Time On-Device (future)

- [ ] **5.1** DPDFNet TFLite for streaming denoise
- [ ] **5.2** Pedalboard real-time DSP chain
- [ ] **5.3** System audio filter integration (CoreAudio on Mac, PipeWire on Linux)
- [ ] **5.4** Latency optimization — target <50ms end-to-end

## Dependency Graph

```
Sprint 1 (benchmark)
    │
    ▼
Sprint 2.1 (restructure) ──────────────────────────┐
    │                                               │
    ├──→ 2.2 (public API)                           │
    │       │                                       │
    ├──→ 2.3 (pyproject) ──┐                        │
    │                      │                        │
    ├──→ 2.4 (python ver) ─┤                        │
    │                      │                        │
    ├──→ 2.5 (tests) ──────┤     Sprint 3.3 (metrics)
    │                      │         │
    │                      ▼         │
    │                  2.6 (GitHub)  │
    │                      │         │
    │                      ▼         │
    │                  2.7 (PyPI) ───┤
    │                      │         │
    │                      ▼         ▼
    │                  3.1 (Space)  3.4 (CLI feedback)
    │                      │
    │                      ▼
    │                  3.2 (feedback widget)
    │
    └──→ 3.3 (metrics) — can start in parallel with 2.2-2.5
```

## Research Findings (2026-04-01)

| Finding | Impact | Action |
|---------|--------|--------|
| DPDFNet (ceva-ip/DPDFNet) — DeepFilterNet successor, ONNX+TFLite | Potential denoiser swap, cleaner API, enables real-time | Benchmark in Sprint 1 |
| ClearVoice Numpy2Numpy API (June 2025) | Eliminates temp file I/O in engine.py | Test in Sprint 1 |
| DNSMOS Pro + ClearVoice SpeechScore | Local quality metrics, no Azure | Add in Sprint 3.3 |
| No full-pipeline competitors exist | Market gap confirmed | Ship fast |
| `cleanfeed` available on PyPI + GitHub | Name is clear | Use it |
