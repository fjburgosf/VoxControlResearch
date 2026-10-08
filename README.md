# VoxControlResearch

**Scientific software for uncertainty-aware voice intent recognition and adaptive command execution.**
Author: Francisco Javier Burgos Flórez · Version 1.0.0

VoxControlResearch is a framework for *studying* how a voice-control system should act when a command is
clear, ambiguous, incomplete, unknown, corrupted by speech-recognition errors, or dependent on context.
Instead of executing the most probable intent blindly, it estimates calibrated uncertainty and the cost of a
wrong action and decides to **EXECUTE**, **CONFIRM** or **REJECT / CLARIFY**. Every experiment runs on a
simulated desktop (sandbox) and can be repeated from text, audio files or datasets. No cloud service or
language-model API is used. The text mode, the text experiments and nine of the ten examples work offline and
without a microphone. The voice features need the speech recognition model, which is downloaded once on first
use, and example 02 and the audio experiment E5 also need the Windows text-to-speech engine (SAPI).

## Scientific contribution: UCIL

**UCIL — Uncertainty-Calibrated Incremental Intent Learning** combines

1. an intent predictor $p(y\mid x)$ (local TF-IDF logistic regression, other predictors pluggable)
2. uncertainty signals: entropy, margin, max-probability, ASR word confidence, context shift
3. OOD detection: MSP, energy, kNN, prototype, Mahalanobis, lexical, and a learned fusion of them
4. calibration: temperature scaling plus an estimator of $P(\text{correct execution}\mid x)$ selected by
   cross-validated Brier score
5. risk-aware routing: $d=\arg\min_d E[C(d)\mid x]$ with an action-dependent error cost
6. incremental adaptation: a per-user correction memory kept separate from the frozen global model.

Every component can be switched off for ablation. The formulation, data and metrics are described in the
technical manual.

**Research question.** Can calibrated uncertainty, OOD detection and incremental learning reduce incorrect
executions and unnecessary confirmations compared with argmax or fixed-threshold decisions? The software is
built to test this, not to assume it.

## Main results (10 evaluation seeds, mean [95% CI])

Synthetic Spanish/English corpus, 17 intents, test paraphrases never seen in training. Costs: wrong
execution 2 / 5 / 20 (low / medium / high risk), confirmation 0.3, rejection 1.

| Method | Mean cost per command | Incorrect execution rate | High-risk incorrect execution rate (count over 10 seeds) | Clarification rate | Rejection rate |
|---|---|---|---|---|---|
| B1 rules | 0.744 [0.675, 0.813] | 0.037 | 0.0073 (63) | 0.002 | 0.642 |
| B2 argmax | 2.674 [2.487, 2.861] | 0.430 | 0.0891 (773) | 0.000 | 0.000 |
| B3 fixed threshold (0.7) | 0.602 [0.544, 0.659] | 0.030 | 0.0020 (17) | 0.727 | 0.000 |
| B4 calibrated threshold | 0.609 [0.507, 0.710] | 0.033 | 0.0036 (31) | 0.497 | 0.235 |
| **UCIL** | **0.563 [0.522, 0.604]** | 0.058 | 0.0003 (3) | 0.442 | 0.254 |

What the experiments show, including where UCIL does **not** win:

* **Cost.** UCIL has a lower mean cost than B3 in 10/10 seeds (paired difference −0.039 [−0.070, −0.007]).
  Against B4 the clean-data difference is not significant (−0.046 [−0.121, 0.029]). Under combined shift
  (paraphrase + 15 % simulated ASR errors + OOD + new users, E10) UCIL is lower than B4 in 10/10 seeds
  (−0.063 [−0.084, −0.043]).
* **Unweighted incorrect executions.** UCIL executes *more* wrong low-risk commands than B3/B4
  (+0.026 to +0.029 in clean data, 0/10 seeds better). It spends interventions on expensive actions:
  its clarification rate rises with risk (low 0.37, medium 0.57, high 0.71) and it executes the fewest
  high-risk commands wrongly (3 of 8673 commands over the 10 seeds, against 17 for B3 and 31 for B4, not zero),
  whereas B3/B4 do not order their interventions by risk.
* **Calibration.** Temperature scaling lowers in-domain ECE from 0.131 to 0.070. UCIL's estimate of
  P(correct execution) has ECE 0.081 on in-domain + OOD data, comparable to temperature scaling (0.075),
  not better. Its risk–coverage ranking (AURC 0.203) equals that of raw confidence (0.201).
* **OOD.** The learned fusion of detectors reaches AUROC 0.816 (far OOD 0.886, near OOD 0.752), above the
  best single detector (lexical, 0.777). Near-OOD requests such as "open the garage door" remain hard.
* **Ablation.** Removing OOD detection raises the incorrect execution rate (0.058 → 0.082). Removing the
  calibration layer or the risk router leaves the mean cost unchanged in this setting, and removing adaptation
  raises the cost under combined shift (0.593 → 0.637).
* **Incremental learning (E7).** The correction memory reaches 0.996 accuracy on new expressions after 102
  corrections with no forgetting. Naive fine-tuning forgets 0.459 of previous accuracy, replay strategies
  0.158–0.214.
* **Personalisation (E8).** After 24 corrections a user's accuracy rises from 0.499 to 0.994 while global
  and other users' accuracy stay unchanged. Replay fine-tuning reaches 0.668 and lowers other users'
  accuracy from 0.497 to 0.381.
* **Ambiguity (E3).** The context prior raises accuracy on context-dependent commands from 0.274 to 0.408.
  With an ambiguous target UCIL asks with explicit options in 100 % of cases (B4: 39 %).

Figures and tables are generated in `results/EXP-YYYY-NNNNNN/` (see *Reproducibility*).

## Windows application

Unzip `VoxControlResearch_1.0.0_Windows_x64.zip` and open `VoxControlResearch\VoxControlResearch.exe`.
No installation or Python is needed. Results, trained models and the speech-model cache are written inside
the application folder (or in `Documents\VoxControlResearch` if that folder is read-only).
`VoxControlResearch.exe --selftest` writes `selftest_report.txt`. It checks the core and the safety
postconditions of the simulated desktop, the ten examples (example 02 with real synthetic speech and speech
recognition), reading WAV and FLAC files (mono and stereo), a reduced run of each text experiment type (main, ambiguity, asr_text_noise, incremental,
combined_shift, personalization, ablation) and every tab, button, language and tutorial step of the interface.
It does not run the audio experiment E5, and in the interface part the voice tab uses a simulated recogniser.

## Installation from source (Windows 10/11, Python 3.11 or 3.12, 64-bit)

From the folder of the source code:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

Speech recognition uses `faster-whisper`, and models are downloaded on first use. Without it the software
runs in text mode and shows "Speech recognition model not available".

## Quick start

```bat
run_voxcontrol.bat                        :: graphical interface
.venv\Scripts\voxcontrol predict "abre el navegador"
.venv\Scripts\voxcontrol example example_03_ambiguity
```

### Graphical interface
Tabs: Home (train/load model, **named examples drop-down**), Text test (intent, calibrated confidence,
OOD score, risk, decision and the actual decision factors, with confirmation in the sandbox or correction of the
system), Voice (microphone or WAV/FLAC, mono or stereo), Adaptation (stored corrections), Experiments, Results
(tables and figures, ZIP export) and Settings (costs and model parameters with symbol and unit, use of the
context prior). The **ES | EN** button switches language without losing the configuration. The **Tutorial**
button opens an interactive guide that walks through a complete experiment and can perform each step.

* When the decision is CONFIRM and there are several options (for example two chat applications), nothing runs
  until one is chosen under *Option to run*. An action whose slots are ambiguous, missing or invalid never runs.
* With *Always confirm high-risk actions* (on by default) a high-risk action is never executed directly.
* Corrections are saved automatically in `models/corrections.json` and are kept after closing the program and
  after retraining. *Delete all corrections* removes them.
* The Settings tab changes only the interactive model. Each experiment uses exclusively the parameters of its
  YAML file, and choosing a file in *Experiments* loads its seeds.
* Load only model files you created: a `.pkl` file of unknown origin can run code.

### Python API
```python
from voxcontrol.api import VoxModel
model = VoxModel.train(seed=0)
r = model.process_text("abre el navegador")
print(r.intent, r.slots, r.confidence, r.calibrated_confidence, r.ood_score, r.decision, r.options)
res = model.execute(r, confirmed=True)   # with several options also pass choice=<the option the user chose>
print(res.success, res.message)          # always on the simulated desktop
model.correct(r, correct_intent="open_app", user="ana")
model.save_corrections("corrections.json")
model.process_audio("command.wav")
```

### CLI
```bat
voxcontrol predict "abre el navegador"
voxcontrol audio command.wav
voxcontrol benchmark configs\exp_main.yaml
voxcontrol calibrate configs\exp_main.yaml --out model.pkl
voxcontrol analyze results\EXP-YYYY-NNNNNN
voxcontrol example                      :: lists the ten examples
voxcontrol gui
```

## Experiments

| Config | Experiments |
|---|---|
| `configs/exp_main.yaml` | E1 recognition/calibration/risk–coverage, E2 paraphrase difficulty, E4 OOD, E9 risk |
| `configs/exp_ambiguity.yaml` | E3 ambiguity and context |
| `configs/exp_audio_noise.yaml` | E5 noisy speech (SNR → WER → intent → decision) |
| `configs/exp_asr_text_noise.yaml` | E6 simulated transcription errors |
| `configs/exp_incremental.yaml` | E7 incremental adaptation and forgetting |
| `configs/exp_personalization.yaml` | E8 per-user adaptation |
| `configs/exp_combined_shift.yaml` | E10 combined shift |
| `configs/exp_ablation.yaml` | ablation of UCIL components |

## Reproducibility
Each run creates `results/EXP-YYYY-NNNNNN/` with `config.yaml`, `metadata.json` (software version, seeds,
library versions, platform), per-seed CSV tables, `summary_*.csv` (mean, SD, median, IQR, 95% CI),
`comparisons_*.csv` (paired differences, Cohen's d_z, Wilcoxon) after `voxcontrol analyze`, figures in
PNG/SVG/PDF and `logs/run.log`. Runs are deterministic for a given configuration and seed. With a single
seed the SD and the 95% CI are not estimable and are left empty (shown as NA in the application). The audio
experiment also writes `audio_environment.csv` (TTS voice, SHA-256 of the synthetic recordings, speech
recognition model with the SHA-256 of its weights, device and library versions): its results are identical
only with the same voice, recordings and recognition model.

## Reproducible examples

Each example repeats an experiment of the user manual. The figures below are what the run reports
(deterministic for the given configuration and seeds). In the application: *Experiments* tab, choose the
configuration, keep the seeds of the file, *Run experiment*, then open the table in *Results*.
From source: `voxcontrol benchmark <config>`.

| Configuration | Table / row | Value reported |
|---|---|---|
| `configs/exp_main.yaml` (seeds 100–109) | `summary_decision_metrics`, UCIL, `mean_cost` | mean 0.563, 95% CI [0.522, 0.604] |
| `configs/exp_main.yaml` | `summary_decision_metrics`, B3_fixed_threshold, `mean_cost` | mean 0.602 |
| `configs/exp_main.yaml` | `summary_decision_metrics`, UCIL, `high_risk_incorrect_execution_rate` | mean 0.000346 |
| `configs/exp_main.yaml` | `summary_ood_detection`, fused, all, `auroc` | mean 0.816 |
| `configs/exp_incremental.yaml` | `summary_incremental`, memory, 102 corrections, `acc_new_expressions` | mean 0.996 |
| `configs/exp_incremental.yaml` | `summary_incremental`, finetune_naive, 102 corrections, `forgetting` | mean 0.459 |
| `configs/exp_personalization.yaml` | `summary_personalization`, ucil_memory, 24 corrections, `user_accuracy` | mean 0.994 |
| `configs/exp_combined_shift.yaml` | `summary_combined_shift`, UCIL, `mean_cost` | mean 0.593 |
| `configs/exp_audio_noise.yaml` (seeds 100–102) | `summary_audio_noise`, UCIL, SNR 0 dB, `mean_cost` | mean 0.709 |

## Building the executable

```bat
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\python.exe tools\build_exe.py
```

The script runs PyInstaller with `packaging/VoxControlResearch.spec`, copies `configs/` and `README.md`,
checks that no internal path exceeds 120 characters and writes
`entregables/VoxControlResearch_<version>_Windows_x64.zip`.

## Tests
```bat
.venv\Scripts\python.exe -m pytest -q
```
Unit tests, scientific sanity tests (clear vs ambiguous uncertainty, OOD scores, risk-dependent thresholds,
non-decreasing personal accuracy after consistent corrections, measured calibration), safety and
reproducibility tests (no action without a chosen target, unresolved slots never run, mandatory confirmation
in the sandbox, persistent corrections, seeds of the YAML file, no interval with one seed, effective B3
threshold, context prior, WAV and FLAC files) and GUI tests.

## Safety
The software never acts on the operating system: every action only changes a simulated desktop, and there
is no real-execution mode. Commands are mapped to a whitelist of validated actions with typed slots, an action
with an ambiguous, missing or invalid slot never runs, and no generated shell command is ever executed.
High-risk actions require confirmation by default.

## Limitations
Text data are synthetic and written from templates. Audio experiments use synthetic English speech with white
noise. Cost values are design parameters, and the local intent model is linear.

## Contact
Francisco Javier Burgos Flórez · fjburgosf@gmail.com
