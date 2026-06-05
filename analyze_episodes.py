"""
Descriptive analysis of medication episode timelines produced by episode_pipe.py.

Reads results/episodes.csv and prints:
  - data quality checks
  - patient-level summary (episode counts, observation span, drug diversity)
  - episode duration distribution
  - medication frequency and treatment-line breakdown
  - dose statistics per medication
"""

import pandas as pd
import numpy as np

INPUT = 'results/episodes.csv'

df = pd.read_csv(INPUT)
df['date']     = pd.to_datetime(df['date'],     errors='coerce')
df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
PLAUSIBLE_START = pd.Timestamp('1900-01-01')
PLAUSIBLE_END   = pd.Timestamp('2100-01-01')

bad_start = df['date']     < PLAUSIBLE_START
bad_end   = df['end_date'] > PLAUSIBLE_END
bad_order = df['end_date'] < df['date']

print("=" * 60)
print("DATA QUALITY")
print("=" * 60)
print(f"Total rows             : {len(df):,}")
print(f"Unique patients        : {df['patient_id'].nunique():,}")
print(f"Missing end_date       : {df['end_date'].isna().sum():,}")
print(f"Missing dose           : {df['dose'].isna().sum():,}")
print(f"Implausible date       : {bad_start.sum():,}  (before {PLAUSIBLE_START.date()})")
print(f"Implausible end_date   : {bad_end.sum():,}  (after {PLAUSIBLE_END.date()})")
print(f"end_date < date        : {bad_order.sum():,}")

clean = df[~bad_start & ~bad_end & df['end_date'].notna()].copy()
clean['duration_days'] = (clean['end_date'] - clean['date']).dt.days
print(f"\nRows after quality filter: {len(clean):,}  "
      f"({len(clean)/len(df)*100:.1f}% of total)\n")

# ---------------------------------------------------------------------------
# Patient-level summary
# ---------------------------------------------------------------------------
pt = clean.groupby('patient_id').agg(
    n_episodes =('order',      'count'),
    first_date =('date',       'min'),
    last_date  =('end_date',   'max'),
    n_drugs    =('medication', 'nunique'),
).copy()
pt['obs_span_days'] = (pt['last_date'] - pt['first_date']).dt.days

print("=" * 60)
print("PATIENTS")
print("=" * 60)
print(f"Unique patients          : {pt.shape[0]:,}\n")

print("Episodes per patient:")
print(pt['n_episodes'].describe().rename({
    'count': 'n_patients', 'mean': 'mean', '50%': 'median',
    'min': 'min', 'max': 'max'
}).to_string())
print()

ep_dist = pt['n_episodes'].value_counts().sort_index()
print("Distribution of episode counts:")
for k, v in ep_dist.items():
    bar = '#' * int(v / ep_dist.max() * 40)
    print(f"  {k:>3} episode(s): {v:>5,}  {bar}")

# ---------------------------------------------------------------------------
# Observation span (time in data)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("OBSERVATION SPAN (per patient, years)")
print("=" * 60)
span_yrs = pt['obs_span_days'] / 365.25
print(span_yrs.describe().rename({
    'count': 'n_patients', 'mean': 'mean', '50%': 'median'
}).apply(lambda x: round(x, 2)).to_string())

# ---------------------------------------------------------------------------
# Episode duration
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EPISODE DURATION (days)")
print("=" * 60)
print(clean['duration_days'].describe().rename({
    'count': 'n_episodes', 'mean': 'mean', '50%': 'median'
}).apply(lambda x: round(x, 1)).to_string())
short = (clean['duration_days'] < 14).sum()
print(f"\nEpisodes < 14 days       : {short:,}  ({short/len(clean)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Medication distribution
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MEDICATION DISTRIBUTION (episodes)")
print("=" * 60)
med_counts = clean['medication'].value_counts()
med_pct    = (med_counts / len(clean) * 100).round(1)
for med, cnt in med_counts.items():
    print(f"  {med:<35} {cnt:>5,}  ({med_pct[med]:.1f}%)")

# ---------------------------------------------------------------------------
# Treatment line (episode order)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TREATMENT LINE — TOP MEDICATION BY EPISODE ORDER")
print("=" * 60)
max_order = min(clean['order'].max(), 7)
for order in range(max_order + 1):
    subset = clean[clean['order'] == order]
    if len(subset) == 0:
        continue
    top = subset['medication'].value_counts().head(3)
    top_str = ', '.join(f"{m} ({n})" for m, n in top.items())
    print(f"  Line {order+1} (n={len(subset):,}): {top_str}")

# ---------------------------------------------------------------------------
# Dose by medication
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DOSE (mg) BY MEDICATION — median  [IQR]")
print("=" * 60)
dose_stats = (
    clean.dropna(subset=['dose'])
    .groupby('medication')['dose']
    .agg(n='count', median='median',
         q25=lambda x: x.quantile(0.25),
         q75=lambda x: x.quantile(0.75))
    .sort_values('n', ascending=False)
)
for med, row in dose_stats.iterrows():
    print(f"  {med:<35} n={int(row['n']):>5,}  "
          f"median={row['median']:>8.1f}  IQR [{row['q25']:.1f}–{row['q75']:.1f}]")

# ---------------------------------------------------------------------------
# Distinct drug count per patient
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("UNIQUE DRUGS PER PATIENT")
print("=" * 60)
print(pt['n_drugs'].describe().rename({
    'count': 'n_patients', '50%': 'median'
}).apply(lambda x: round(x, 1)).to_string())
print()
for k, v in pt['n_drugs'].value_counts().sort_index().items():
    bar = '#' * int(v / pt['n_drugs'].value_counts().max() * 40)
    print(f"  {k:>2} drug(s): {v:>5,}  {bar}")

print("\nDone.")
