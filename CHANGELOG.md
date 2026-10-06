# Changelog

## VoxControlResearch 1.0.0 — 2026-10-06
First release of the scientific core (`src/voxcontrol/`).

### Added
- UCIL: intent predictor, uncertainty estimator, OOD detector (individual detectors and a learned fusion),
  calibrator selected by cross-validated Brier score, risk-aware router, per-user correction memory.
- Baselines B1 rules, B2 argmax, B3 fixed threshold, B4 calibrated threshold.
- Extensible YAML intent registry (17 intents, ES/EN, risk levels, slots, graded paraphrase templates).
- Reproducible synthetic corpus with template-disjoint splits, near/far OOD topics, ambiguity sets and
  simulated users; text-level ASR error simulation.
- Audio processing and controlled degradations; faster-whisper recogniser with word-probability uncertainty.
- Slot extraction and validation; simulated desktop sandbox with an action whitelist.
- Experiments E1–E10 and ablation, multi-seed runner, summaries with 95% intervals, paired comparisons,
  figures (PNG/SVG/PDF), experiment records `EXP-YYYY-NNNNNN`.
- Python API (`VoxModel`), CLI (`voxcontrol predict | audio | benchmark | calibrate | analyze | example | gui`).
- Bilingual GUI (ES/EN) with named-examples menu and interactive tutorial.
- Scientific sanity tests A–E, unit tests and GUI tests.
