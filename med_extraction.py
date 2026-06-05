"""
Antipsychotic medication extraction from free clinical text using medspacy.

Loads gazetteer rules from 'antipsychotics_lookup.json' and applies them to
clinical notes. Output can be combined with structured prescription tables
to build the full medication timeline for episode detection.

Usage
-----
    python med_extraction.py --input notes.csv --text_col text --id_col doc_id

    Or import and call extract_medications() directly.
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import medspacy
import pandas as pd
from medspacy.ner import TargetRule

LOOKUP_PATH = Path(__file__).parent / "antipsychotics_lookup.json"
CACHE_PATH  = Path(__file__).parent / "data" / "med_nlp_cache.pkl"


# ---------------------------------------------------------------------------
# Build medspacy pipeline
# ---------------------------------------------------------------------------

def _build_rules(lookup_path: Path) -> tuple[list[TargetRule], dict]:
    """Build TargetRules from the gazetteer and return (rules, gazetteer_map).

    gazetteer_map: lowercased gazetteer term -> {drug_name, AP_type}
    """
    import json
    with open(lookup_path) as fh:
        entries = json.load(fh)

    gazetteer_map = {
        e["gazetteer"].lower().strip(): {
            "drug_name": e["drug_name"],
            "AP_type":        e["AP_type"],
        }
        for e in entries
    }

    meds = list(gazetteer_map.keys())

    # Multi-token terms whose surface form contains a shorter term as a prefix
    # need token-level patterns so medspacy matches them greedily.
    links: dict[str, list[str]] = {}
    for i, mi in enumerate(meds):
        for mj in meds[i + 1:]:
            if mi + " " in mj:
                links.setdefault(mj, []).append(mi)

    independent = set(meds) - set(links) - {t for vs in links.values() for t in vs}

    rules = []
    for term in independent:
        rules.append(TargetRule(term, "Antipsychotic"))
    for term, parts in links.items():
        pattern = [{"LOWER": p} for p in parts]
        rules.append(TargetRule(term, "Antipsychotic", pattern=pattern))

    return rules, gazetteer_map


def build_nlp(lookup_path: Path = LOOKUP_PATH, cache_path: Optional[Path] = CACHE_PATH):
    """Return a medspacy NLP pipeline loaded with antipsychotic target rules.

    The pipeline is cached to disk after the first build to speed up repeated
    calls. Pass cache_path=None to disable caching.
    """
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as fh:
            nlp = pickle.load(fh)
        return nlp

    rules, _ = _build_rules(lookup_path)

    nlp = medspacy.load()
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(rules)

    if cache_path:
        with open(cache_path, "wb") as fh:
            pickle.dump(nlp, fh)

    return nlp


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_medications(
    texts: pd.DataFrame,
    id_col: str = "doc_id",
    text_col: str = "text",
    lookup_path: Path = LOOKUP_PATH,
    nlp=None,
) -> pd.DataFrame:
    """Extract antipsychotic mentions from clinical text.

    Parameters
    ----------
    texts : DataFrame with at least `id_col` and `text_col` columns.
    id_col : column name for document / patient identifier.
    text_col : column name containing free text.
    lookup_path : path to 'antipsychotics_lookup.json'.
    nlp : pre-built medspacy pipeline (built once and reused if provided).

    Returns
    -------
    DataFrame with columns:
        doc_id, start, end, surface_form, drug_name, AP_type
    """
    if nlp is None:
        nlp = build_nlp(lookup_path)

    _, gazetteer_map = _build_rules(lookup_path)

    rows = []
    for _, record in texts.iterrows():
        doc_id = record[id_col]
        text   = str(record[text_col])
        doc    = nlp(text.lower())

        for ent in doc.ents:
            surface = text[ent.start_char:ent.end_char]
            meta    = gazetteer_map.get(ent.text, {})
            rows.append({
                "doc_id":         doc_id,
                "start":          ent.start_char,
                "end":            ent.end_char,
                "surface_form":   surface,
                "drug_name": meta.get("drug_name", ent.text),
                "AP_type":        meta.get("AP_type", ""),
            })

    return pd.DataFrame(rows, columns=[
        "doc_id", "start", "end", "surface_form", "drug_name", "AP_type",
    ])


# ---------------------------------------------------------------------------
# Combine NLP output with structured prescription data
# ---------------------------------------------------------------------------

def combine_with_structured(
    nlp_df: pd.DataFrame,
    structured_df: pd.DataFrame,
    nlp_date_col: Optional[str] = None,
    struct_date_col: str = "date",
    struct_id_col: str = "patient_id",
) -> pd.DataFrame:
    """Merge NLP-extracted mentions with structured prescription rows.

    Both sources are normalised to the same schema and concatenated so the
    result can be fed directly into episode_pipe.py.

    Parameters
    ----------
    nlp_df : output of extract_medications(), must contain 'doc_id' mapped to
             patient IDs (or a 'patient_id' column if already resolved).
    structured_df : e.g. antipsychotic_prescriptions.csv loaded as a DataFrame.
    nlp_date_col : date column in nlp_df (optional; rows without dates get NaT).
    struct_date_col : date column in structured_df.
    struct_id_col : patient-ID column in structured_df.

    Returns
    -------
    DataFrame with columns: patient_id, date, drug_name, AP_type, source
    """
    # --- NLP side ---
    nlp_out = nlp_df.copy()
    nlp_out["source"] = "nlp_text"
    nlp_out["date"] = (
        pd.to_datetime(nlp_out[nlp_date_col], errors="coerce")
        if nlp_date_col and nlp_date_col in nlp_out.columns
        else pd.NaT
    )
    id_col = "patient_id" if "patient_id" in nlp_out.columns else "doc_id"
    nlp_out = nlp_out.rename(columns={id_col: "patient_id"})[
        ["patient_id", "date", "drug_name", "AP_type", "source"]
    ]

    # --- Structured side ---
    struct_out = structured_df.copy()
    struct_out["source"] = "structured"
    struct_out["date"] = pd.to_datetime(
        struct_out[struct_date_col], errors="coerce"
    )
    struct_out["drug_name"] = (
        struct_out["drug_name"].str.lower().str.strip()
    )
    if "AP_type" not in struct_out.columns:
        struct_out["AP_type"] = ""
    struct_out = struct_out.rename(columns={struct_id_col: "patient_id"})[
        ["patient_id", "date", "drug_name", "AP_type", "source"]
    ]

    return pd.concat([struct_out, nlp_out], ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Extract antipsychotic medications from clinical text.")
    p.add_argument("--input",     required=True, help="CSV file with clinical notes")
    p.add_argument("--output",    default="data/med_extracted.csv", help="Output CSV path")
    p.add_argument("--text_col",  default="text",   help="Column containing free text")
    p.add_argument("--id_col",    default="doc_id", help="Column for document/patient ID")
    p.add_argument("--lookup",    default=str(LOOKUP_PATH), help="Path to Antipsychotics Look Up.xlsx")
    p.add_argument("--no_cache",  action="store_true", help="Disable NLP pipeline caching")
    return p.parse_args()


def main():
    args = _parse_args()

    print(f"Loading notes from {args.input} ...")
    notes = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(notes):,} documents")

    cache = None if args.no_cache else CACHE_PATH
    nlp   = build_nlp(Path(args.lookup), cache_path=cache)

    print("Extracting antipsychotic mentions ...")
    results = extract_medications(
        notes,
        id_col=args.id_col,
        text_col=args.text_col,
        lookup_path=Path(args.lookup),
        nlp=nlp,
    )

    print(f"  {len(results):,} mentions found across {results['doc_id'].nunique():,} documents")
    print(results["drug_name"].value_counts().head(10))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
