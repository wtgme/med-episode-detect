# Medication Episode Detection

[![GitHub](https://img.shields.io/badge/GitHub-med--episode--detect-blue?logo=github)](https://github.com/wtgme/med-episode-detect)

Detects medication episodes from structured prescription records and optional NLP extraction from clinical text. Uses PELT change-point detection to segment each patient's prescription history into discrete treatment episodes, then identifies polypharmacy periods around transitions.

---

## Quick start

### 1. Install dependencies

Core pipeline:
```bash
pip install -r requirements.txt
```

NLP extraction (optional):
```bash
pip install medspacy
```

### 2. Run the full pipeline on synthetic sample data
```bash
python run_pipeline.py
```

This generates 60 synthetic patients, detects episodes, prints statistics, and produces an interactive HTML timeline viewer.

### 3. Open the viewer
Open `results/viewer.html` in any browser to explore the timeline interactively.

---

## Pipeline scripts

| Script | Description |
|---|---|
| `generate_sample_data.py` | Creates `data/antipsychotic_prescriptions.csv` with synthetic patients for testing. Replace with your own data (same column structure) to run on real records. |
| `episode_pipe.py` | Detects medication episodes using PELT change-point detection. |
| `analyze_episodes.py` | Prints descriptive statistics for the detected episodes. |
| `visualize_plotly.py` | Generates a self-contained interactive HTML timeline viewer. |
| `med_extraction.py` | *(Optional)* Extracts medication mentions from free clinical text using medspacy. Output can be merged with structured records via `combine_with_structured()`. |
| `test_med_extraction.py` | End-to-end test for the NLP extraction pipeline. |

### Inputs and outputs

```
episode_pipe.py
    Input  : data/antipsychotic_prescriptions.csv
    Output : results/episodes.csv
             results/episodes_sequences.pkl
             results/episodes_polypharmacy.pkl

analyze_episodes.py
    Input  : results/episodes.csv

visualize_plotly.py
    Input  : data/antipsychotic_prescriptions.csv  (cached to data/aps_cache.pkl)
             results/episodes.csv
    Output : results/viewer.html

med_extraction.py
    Usage  : python med_extraction.py --input notes.csv --id_col patient_id
```

---

## Input data format

`antipsychotic_prescriptions.csv` must contain at minimum:

| Column | Type | Description |
|---|---|---|
| `patient_id` | integer | Patient identifier |
| `drug_name` | string | Normalised drug name (e.g. `olanzapine`) |
| `date` | date | Prescription date (`YYYY-MM-DD`) |
| `dose` | string | Dose string (e.g. `10mg`, `200mg`) |
| `source_table` | string | Source system identifier; rows whose `source_table` starts with `supplementary` are retained regardless of dose format |

Extra columns are ignored.

---

## `run_pipeline.py` flags

| Flag | Description |
|---|---|
| `--regen` | Regenerate sample data even if `data/antipsychotic_prescriptions.csv` exists |
| `--no-viz` | Skip the HTML visualisation step (faster for batch runs) |

## `visualize_plotly.py` flags

| Flag | Description |
|---|---|
| `--n N` | Number of patients to display (default: 20) |
| `--patient_ids ID [ID ...]` | Specific patient IDs to include |
| `--out PATH` | Output HTML path (default: `results/viewer.html`) |
| `--rebuild-cache` | Force rebuild of the APS pickle cache (required when switching datasets) |

---

## NLP extraction

`med_extraction.py` uses medspacy and a gazetteer (`antipsychotics_lookup.json`) to extract antipsychotic mentions from free clinical text. It resolves brand names and common misspellings to normalised drug names.

```python
from med_extraction import build_nlp, extract_medications, combine_with_structured

nlp = build_nlp()
mentions = extract_medications(notes_df, id_col="patient_id", text_col="text", nlp=nlp)
merged = combine_with_structured(nlp_df=mentions, structured_df=rx_df)
```

`combine_with_structured()` returns a DataFrame with columns `patient_id`, `date`, `drug_name`, `AP_type`, `source` that can be fed directly into the episode detection pipeline.

Run the end-to-end NLP test:
```bash
python test_med_extraction.py
```

---

## Extending to other medication classes

The detection algorithm, statistics, and visualisation are fully generic — they operate on `drug_name`, `date`, and `patient_id` regardless of medication class. Only four things need to change.

### 1. Replace the NLP gazetteer — `antipsychotics_lookup.json`

Create a new JSON file with the same structure for your target medication class:

```json
[
  {"gazetteer": "methotrexate",  "drug_name": "Methotrexate", "AP_type": "DMARD"},
  {"gazetteer": "mtx",           "drug_name": "Methotrexate", "AP_type": "DMARD"},
  {"gazetteer": "humira",        "drug_name": "Adalimumab",   "AP_type": "biologic"}
]
```

Each entry maps a surface form (`gazetteer`) — including brand names and misspellings — to a normalised `drug_name`. The `AP_type` field can be repurposed as any classification label (drug class, generation, route, etc.).

> If you are not using NLP extraction (`med_extraction.py`), this file can be ignored entirely.

### 2. Update the gazetteer path in `med_extraction.py`

```python
# Line 24 — point to your new gazetteer file
LOOKUP_PATH = Path(__file__).parent / "antipsychotics_lookup.json"
```

Also update the entity label on line 63 inside `_build_rules()`:

```python
rules.append(TargetRule(term, "Antipsychotic"))   # change label to match your drug class
```

### 3. Update the sample data generator — `generate_sample_data.py`

Replace the `DRUGS` dictionary with your medication names and typical doses:

```python
DRUGS = {
    'methotrexate': [10, 15, 20, 25],
    'adalimumab':   [40],
    'etanercept':   [25, 50],
}
```

### 4. Update the viewer title — `visualize_plotly.py`

Search for `"Antipsychotic Episode Timeline Viewer"` and replace with an appropriate title. This is cosmetic only and does not affect functionality.

### Nothing else needs changing

| Component | Status |
|---|---|
| `episode_pipe.py` — PELT detection, polypharmacy | Fully generic |
| `analyze_episodes.py` — statistics | Fully generic |
| `run_pipeline.py` — orchestration | Fully generic |
| Input CSV format | Fully generic |

---

## Security note

Real patient data must never be committed. `data/` and `results/` are covered by `.gitignore`, with explicit exceptions only for synthetic sample files. See `.gitignore` for the full list of permitted files.
