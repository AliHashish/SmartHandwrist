# PPG-DaLiA · Notebook-first prototype

A **notebook-first**, transparent take-home exercise: explore wrist signals, estimate HR from PPG only, evaluate against ECG-derived reference labels, and demonstrate an evidence-grounded agent. Built for clarity and auditability, not clinical use.

## Suggested reading order

| Notebook | Core question | Output |
|---|---|---|
| `notebooks/01_eda_visualization.ipynb` | What do the wrist signals and labels look like? | Data validation, raw BVP/ACC and reference histograms |
| `notebooks/02_heart_rate_estimation.ipynb` | Can a simple PPG frequency method estimate HR? | Raw vs filtered PPG, spectrum, HR predictions, MAE |
| `notebooks/03_evaluation_error_analysis.ipynb` | When do the predictions fail? | Cross-segment MAE, errors and ACC context |
| `notebooks/04_ai_agent.ipynb` | Can a tool-calling AI explain real computed results? | Bounded analysis tool, deterministic fallback, optional Ollama |

`ppg_demo/` contains reusable Python code (loading, estimation, evaluation, agent). `tests/` contains synthetic invariant checks. `REPORT.md` is the required short report **template**; fill it only with your own executed findings. There is **no pretrained ML model or training phase**: notebook 02 builds the requested signal-processing baseline.

## 1 · Install

Python 3.10+ recommended. From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppg-dalia --display-name 'Python (PPG-DaLiA)'
```

## 2 · Data files (never commit them)

Get the **official** archive at https://archive.ics.uci.edu/dataset/495/ppg-dalia . Extract the nested `data.zip` (if present) so that the files look like:

```text
ppg_dalia_notebook_project/
├── data/                     # gitignored
│   └── PPG_FieldStudy/
│       ├── S1/S1.pkl
│       ├── S2/S2.pkl
│       └── ...
├── notebooks/
├── ppg_demo/
├── tests/
└── REPORT.md
```

You can instead set `PPG_DATA_ROOT` to any existing directory containing `S1/S1.pkl`, `S2/S2.pkl`, etc. The examples use S1 and default starts of 30 and 900 seconds; edit these if appropriate after seeing your recording.

**Security:** Official synchronized records are Python pickles, which can run code during deserialization. Only load the trusted official download in a sufficiently isolated environment. The local agent accepts only `S1`–`S15` and bounded time windows, never arbitrary pickle paths. The dataset is not bundled here.

## 3 · Run the notebooks in order

```bash
jupyter lab
```

Run each notebook **top to bottom** and inspect the actual data. No API keys are required. Each notebook contains visible parameters and expected checks.

**Optional synthetic-only smoke tests when you have not downloaded UCI:**

```bash
# Linux/macOS
PPG_DEMO_MODE=1 jupyter lab
# Windows PowerShell
$env:PPG_DEMO_MODE='1'; jupyter lab
```

Synthetic samples are intentionally artificial, for **code checking only**, and must not be submitted or written as real-dataset results. Normal/default mode requires the real UCI dataset.

## 4 · Optional actual LLM agent, free and local

The `04_ai_agent.ipynb` notebook runs a deterministic result explanation by default. That fallback is **not** an LLM and does not fulfill a claim of having run an actual AI agent. For a genuine tool-calling agent:

1. Install [Ollama](https://ollama.com/) and start it.
2. `ollama pull qwen3:4b` (or a tool-calling alternative).
3. With **real** UCI data available, launch Jupyter with `PPG_RUN_OLLAMA=1`.
4. Run the cells in notebook 04 and verify the answer against the printed JSON.

The implementation is in `ppg_demo/agent.py`, using only Python's standard `urllib` for local Ollama API calls. The LLM first asks for `analyze_segment`, gets a bounded JSON summary, and must explain the data without medical diagnosis. The local LLM path is provided but **not verified** in the authoring environment.

## 5 · Verify invariants and fill report

```bash
python -m pytest -q
```

The tests include a **reference-leakage invariant**: modifying reference HR labels cannot affect PPG estimates, only evaluation metrics. That test uses synthetic waveforms, not downloaded recordings. Fill `REPORT.md` with your **real** observations, results (including unsuccessful attempts), limitations, resources and AI tools used, how you verified their outputs, and honest time spent. Neither source data nor keys belong in GitHub or the handoff ZIP.

## Method notes

- Wrist BVP sampled at **64 Hz**, ACC at **32 Hz**. 8-second/2-second windows align with ECG-derived `label` indices.
- Method A: raw PPG periodogram. Method B: Butterworth 0.65–3.5 Hz bandpass followed by Welch spectral peak; `BPM = 60 * f_peak`.
- The HR reference is **never** an estimation input; it is used only to measure errors. Wrist acceleration provides **context indicators** only, not motion correction.
- Since `nperseg` is set to the entire 8-second window, Welch and periodogram are both effectively one-window spectral estimates with the same Hann window; the useful A/B difference here is primarily **bandpass preprocessing**, not an independent estimator family.
- Zero-padding 4096-point FFT improves display interpolation, not intrinsic spectral resolution.
- A short single-subject analysis, repeated-overlapping windows, heuristic motion indicators and sensor artifacts limit generalization. The output is **not a medical diagnosis**.

## Sources

- [UCI PPG-DaLiA dataset](https://archive.ics.uci.edu/dataset/495/ppg-dalia), DOI: [10.24432/C53890](https://doi.org/10.24432/C53890).
- Reiss et al. (2019), [*Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks*](https://doi.org/10.3390/s19143079).
- [Official UCI README](https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf).
- [Python pickle warning](https://docs.python.org/3/library/pickle.html).
- [Ollama tool-calling guide](https://docs.ollama.com/capabilities/tool-calling).

## AI usage disclosure

This starter code, notebook scaffolding and reporting template were prepared with ChatGPT assistance. Real dataset findings, experimental choices, local execution, agent statements, failures and elapsed time **must be personally verified** and documented before submission.
