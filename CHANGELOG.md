# Changelog

## VoxControlResearch 1.0.0 — release date 2026-10-08
First release of the scientific core (`src/voxcontrol/`).

### Added
- UCIL: intent predictor, uncertainty estimator, OOD detector (individual detectors and a learned fusion),
  calibrator selected by cross-validated Brier score, risk-aware router, per-user correction memory.
- Baselines B1 rules, B2 argmax, B3 fixed threshold, B4 calibrated threshold.
- Extensible YAML intent registry (17 intents, ES/EN, risk levels, slots, graded paraphrase templates).
- Reproducible synthetic corpus with template-disjoint splits, near/far OOD topics, ambiguity sets and
  simulated users, and text-level ASR error simulation.
- Audio processing and controlled degradations, faster-whisper recogniser with word-probability uncertainty.
- Slot extraction and validation, simulated desktop sandbox with an action whitelist.
- Experiments E1–E10 and ablation, multi-seed runner, summaries with 95% intervals, paired comparisons,
  figures (PNG/SVG/PDF), experiment records `EXP-YYYY-NNNNNN`.
- Python API (`VoxModel`), CLI (`voxcontrol predict | audio | benchmark | calibrate | analyze | example | gui`).
- Bilingual GUI (ES/EN) with named-examples menu and interactive tutorial.
- Scientific sanity tests A–E, unit tests and GUI tests.

### Safety and reproducibility (before release, after an internal review)
- An action with several options (for example two chat applications) runs only after the user chooses one,
  and only the chosen target changes. An action with an ambiguous, missing or invalid slot never runs, and the
  simulated desktop no longer closes the active window when no application is given.
- Mandatory confirmation of high-risk actions applies on the simulated desktop. The unused real-execution flag
  was removed: every action is simulated.
- Corrections are saved automatically (`models/corrections.json`) and survive restarts and retraining.
- The Experiments tab loads the seeds of the selected YAML file and shows how many runs will be executed.
  With one seed the SD and the 95% CI are reported as not estimable instead of a zero-width interval.
- The Settings tab gained the context-prior switch and lost the B3 threshold, which belongs to the experiment
  files. A warning is shown before loading a model file.
- FLAC files (mono and stereo) are read with `soundfile`, now a declared dependency included in the application.
- The audio experiment records its environment (TTS voice, SHA-256 of the recordings, speech model and the
  SHA-256 of its weights, device, versions).
- The self-test checks safety postconditions and also runs reduced personalization and ablation experiments.
- New regression tests for all of the above.
