"""
End-to-end test for med_extraction.py.

1. Generates synthetic clinical notes that mention antipsychotic medications.
2. Runs extract_medications() to extract NLP mentions.
3. Combines output with structured prescription data via combine_with_structured().
4. Validates the merged DataFrame is compatible with episode_pipe.py input format.
5. Writes outputs to data/ so they can be inspected.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from med_extraction import extract_medications, combine_with_structured, build_nlp

os.makedirs("data", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Synthetic clinical notes
# ---------------------------------------------------------------------------

NOTES = [
    # patient 1001 — started on olanzapine, switched to risperidone
    {"patient_id": 1001, "note_date": "2015-03-10",
     "text": "Patient commenced on Olanzapine 10mg. Tolerating well."},
    {"patient_id": 1001, "note_date": "2015-06-01",
     "text": "Olanzapine dose increased to 15mg due to persistent symptoms."},
    {"patient_id": 1001, "note_date": "2016-01-15",
     "text": "Decision to switch from olanzapine to risperidone 4mg given weight gain."},
    {"patient_id": 1001, "note_date": "2016-04-20",
     "text": "Patient now established on Risperdal. Mental state stable."},

    # patient 1002 — quetiapine throughout, brand name variant
    {"patient_id": 1002, "note_date": "2017-05-01",
     "text": "Started Quetiapine 200mg for psychotic symptoms."},
    {"patient_id": 1002, "note_date": "2017-11-10",
     "text": "Quetiapine increased to 400mg. Good response noted."},
    {"patient_id": 1002, "note_date": "2018-03-05",
     "text": "Continuing on Seroquel (quetiapine) 400mg. No side effects."},

    # patient 1003 — aripiprazole with misspelling + brand, then haloperidol
    {"patient_id": 1003, "note_date": "2014-09-01",
     "text": "Commenced aripipazole 15mg (Abilify). Review in 3 months."},
    {"patient_id": 1003, "note_date": "2015-02-14",
     "text": "Aripiprazole 20mg. Patient non-compliant. Consider switch."},
    {"patient_id": 1003, "note_date": "2015-08-20",
     "text": "Switched to Haloperidol 5mg depot injection."},
    {"patient_id": 1003, "note_date": "2016-01-10",
     "text": "Continuing haloperidol. Extrapyramidal symptoms managed with procyclidine."},

    # patient 1004 — clozapine, brand name 'Clozaril'
    {"patient_id": 1004, "note_date": "2019-06-01",
     "text": "Initiated Clozaril (clozapine) following two failed trials."},
    {"patient_id": 1004, "note_date": "2019-12-01",
     "text": "Clozapine 350mg. WBC satisfactory. Continuing."},

    # patient 1005 — no medication mention (should produce no NLP rows)
    {"patient_id": 1005, "note_date": "2020-01-01",
     "text": "Patient reviewed. No changes to management plan at this time."},
]

# Assign a unique note_id per row so we can join dates back after extraction
notes_df = pd.DataFrame(NOTES).reset_index().rename(columns={"index": "note_id"})
notes_df.to_csv("data/test_notes.csv", index=False)
print(f"Generated {len(notes_df)} synthetic clinical notes for "
      f"{notes_df['patient_id'].nunique()} patients.\n")


# ---------------------------------------------------------------------------
# 2. NLP extraction  (use note_id as doc key so dates join back unambiguously)
# ---------------------------------------------------------------------------

print("Building NLP pipeline (no cache) ...")
nlp = build_nlp(cache_path=None)

print("Extracting medication mentions ...")
nlp_results = extract_medications(
    notes_df,
    id_col="note_id",      # unique per note → safe join key
    text_col="text",
    nlp=nlp,
)

# Join patient_id and note_date back from the notes table
note_meta = notes_df[["note_id", "patient_id", "note_date"]].rename(
    columns={"note_id": "doc_id", "note_date": "date"}
)
nlp_results = nlp_results.merge(note_meta, on="doc_id", how="left")
nlp_results["date"] = pd.to_datetime(nlp_results["date"], errors="coerce")

# Re-order columns for clarity
nlp_results = nlp_results[
    ["patient_id", "date", "surface_form", "drug_name", "AP_type", "start", "end"]
]

print(f"\nExtracted {len(nlp_results)} medication mentions:")
print(nlp_results[["patient_id", "date", "surface_form", "drug_name", "AP_type"]]
      .to_string(index=False))

nlp_results.to_csv("data/test_nlp_extracted.csv", index=False)
print(f"\nSaved → data/test_nlp_extracted.csv")


# ---------------------------------------------------------------------------
# 3. Assertions on NLP output
# ---------------------------------------------------------------------------

errors = []

expected_patients_with_mentions = {1001, 1002, 1003, 1004}
found_patients = set(nlp_results["patient_id"].unique())
missing = expected_patients_with_mentions - found_patients
if missing:
    errors.append(f"Missing patients in NLP output: {missing}")

if 1005 in found_patients:
    errors.append("Patient 1005 should have no NLP mentions but does.")

if nlp_results["drug_name"].isna().any():
    errors.append("Some NLP rows have null drug_name.")
if nlp_results["AP_type"].isna().any():
    errors.append("Some NLP rows have null AP_type.")

drug_names_found = set(nlp_results["drug_name"].str.lower().unique())
for expected in ["olanzapine", "risperidone", "quetiapine", "aripiprazole", "haloperidol"]:
    if expected not in drug_names_found:
        errors.append(f"Expected drug '{expected}' not found in NLP output.")

# Brand 'Risperdal' → 'Risperidone'
risperdal_rows = nlp_results[nlp_results["surface_form"].str.lower() == "risperdal"]
if not risperdal_rows.empty:
    resolved = risperdal_rows["drug_name"].iloc[0].lower()
    if resolved != "risperidone":
        errors.append(f"'Risperdal' resolved to '{resolved}', expected 'risperidone'.")

# Misspelling 'aripipazole' → 'Aripiprazole'
arip_rows = nlp_results[nlp_results["surface_form"].str.lower() == "aripipazole"]
if not arip_rows.empty:
    resolved = arip_rows["drug_name"].iloc[0].lower()
    if resolved != "aripiprazole":
        errors.append(f"'aripipazole' resolved to '{resolved}', expected 'aripiprazole'.")

# Dates must be present for all extracted rows
null_dates_nlp = nlp_results["date"].isna().sum()
if null_dates_nlp > 0:
    errors.append(f"NLP output has {null_dates_nlp} rows with missing dates.")


# ---------------------------------------------------------------------------
# 4. Combine with structured data
# ---------------------------------------------------------------------------

print("\nLoading structured prescription data ...")
structured_df = pd.read_csv("data/antipsychotic_prescriptions.csv")
structured_df = structured_df.rename(columns={
    "brcid": "patient_id", "drug_same_name": "drug_name", "start_date": "date"
}, errors="ignore")

# combine_with_structured expects nlp_df to have a 'doc_id' column (patient_id here)
nlp_for_merge = nlp_results.rename(columns={"patient_id": "doc_id"})

merged = combine_with_structured(
    nlp_df=nlp_for_merge,
    structured_df=structured_df,
    nlp_date_col="date",
    struct_date_col="date",
    struct_id_col="patient_id",
)

print(f"\nMerged DataFrame: {len(merged):,} rows, "
      f"{merged['patient_id'].nunique()} patients")
print(f"Sources: {merged['source'].value_counts().to_dict()}")
print(f"\nColumns: {list(merged.columns)}")
print(f"\nSample (NLP rows):")
print(merged[merged["source"] == "nlp_text"]
      [["patient_id", "date", "drug_name", "AP_type", "source"]]
      .head(8).to_string(index=False))

merged.to_csv("data/test_merged_for_episodes.csv", index=False)
print(f"\nSaved → data/test_merged_for_episodes.csv")


# ---------------------------------------------------------------------------
# 5. Validate merged output is episode-pipe compatible
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"patient_id", "date", "drug_name"}
missing_cols = REQUIRED_COLS - set(merged.columns)
if missing_cols:
    errors.append(f"Merged output missing required columns: {missing_cols}")

if merged["patient_id"].isna().any():
    errors.append("Merged output has null patient_id values.")

date_parsed = pd.to_datetime(merged["date"], errors="coerce")
null_dates_merged = date_parsed.isna().sum()
if null_dates_merged > 0:
    errors.append(f"Merged output has {null_dates_merged} unparseable date values.")

if merged["drug_name"].isna().any():
    errors.append("Merged output has null drug_name values.")

# NLP patients not in the structured data should still appear in the merged output
nlp_only_patients = set(nlp_results["patient_id"]) - set(structured_df["patient_id"])
for pid in nlp_only_patients:
    if pid not in merged["patient_id"].values:
        errors.append(f"Patient {pid} (NLP-only) missing from merged output.")


# ---------------------------------------------------------------------------
# 6. Results
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
if errors:
    print("FAILURES:")
    for e in errors:
        print(f"  FAIL  {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    print(f"  NLP mentions extracted    : {len(nlp_results)}")
    print(f"  Patients with mentions    : {len(found_patients)}")
    print(f"  Brand/misspelling resolve : OK")
    print(f"  Date assignment           : OK (0 nulls)")
    print(f"  Merged rows               : {len(merged):,}")
    print(f"  Episode-pipe compatible   : YES "
          f"(columns: patient_id, date, drug_name)")
