# PPG-DaLiA — Prototype report (fill in after running)

> **Status: template, not a completed experimental report.** Replace bracketed prompts with actual figures from the **official dataset**. Do not include synthetic smoke-test values as observations.

## 1. Dataset and problem understanding

**Dataset:** UCI PPG-DaLiA, 15 participants; wrist BVP 64 Hz, wrist ACC 32 Hz; ECG-derived 8s HR labels / 2s stride.

**Problem:** Estimate HR in BPM using wrist PPG only; use ACC as contextual information, and ECG-derived HR only for evaluation. Motion artifacts make this difficult.

**My observation:** [Describe S1 durations/shapes, actual missingness and one visible signal characteristic. Reference notebook 01.]

## 2. EDA and visualization

[Specify exact participant, selected time, 2–3 concrete findings from raw BVP, ACC, HR-label distribution. Insert or link one/three generated plots if helpful.]

## 3. What I tried and why

- Baseline A: windowed raw BVP spectral peak → BPM.
- Baseline B: 0.65–3.5 Hz Butterworth filter + Welch peak → BPM.
- Windows: 8 s, stride 2 s; reference labels aligned per window; never supplied to the estimator.
- [What changed, what failed, if motion fooled the peak, whether the filter actually helped.]

## 4. Measured results — actual real data only

| Segment (subject / s) | Method | Evaluated windows | MAE (BPM) | Mean signed error (BPM) |
|---|---|---:|---:|---:|
| [S1, 30–150] | Raw periodogram | [ ] | [ ] | [ ] |
| [S1, 30–150] | Filtered + Welch | [ ] | [ ] | [ ] |
| [S1, 900–1020] | Raw periodogram | [ ] | [ ] | [ ] |
| [S1, 900–1020] | Filtered + Welch | [ ] | [ ] | [ ] |

[Summarize actual best/worst windows and any counterexamples. Do **not** present these hand-selected windows as broad generalization.]

## 5. Agent / tool demonstration

- **Actual LLM agent executed?** [Yes/No]. If yes, local model name: [ ]. If no, I demonstrated a deterministic explanation as a fallback; this is *not* an actual LLM.
- **Single tool:** `analyze_segment(subject, start_seconds, duration_seconds)`.
- **Verification:** [Check at least 2 numerical claims in the generated answer against the returned JSON; describe any unsupported claim/error.] 
- **Safety boundary:** agent cannot choose arbitrary files; no diagnosis; reference only used for evaluation.

## 6. What works / what is incomplete

**Works:** [Which notebooks and test commands actually ran, on which subjects?]

**Incomplete / blockers:** [Any failed attempts, performance issues, inability to run local model, file issues. Explain next remedy.]

## 7. Limitations and next steps

- Movement frequencies and harmonics may dominate wrist PPG.
- Two preselected segments from one participant are not a subject-independent validation set; overlapping windows are correlated.
- ACC-overlap / spectral-peak measures are uncalibrated heuristics, not probabilities.
- [Priority next step: more segments and multiple held-out subjects, explicit activity-based sampling, or motion-robust estimation.]

## 8. Resources, AI assistance, verification and time

- Original dataset and README: https://archive.ics.uci.edu/dataset/495/ppg-dalia
- Original paper: https://doi.org/10.3390/s19143079
- Python/SciPy/Matplotlib/NumPy/Pandas documentation, Ollama docs (if used).
- **AI assistance used:** ChatGPT assisted with project scaffolding and initial code. [Add your own prompts/tools, modifications, checks, and how you verified outputs.]
- **Verification:** [Actual test results; code-reviewed estimator/references; checked plots/metrics against raw data; exact model/tool trace if executed.]
- **Approximate time spent:** [Your genuine time, include setup + analysis + debugging + writing.]
