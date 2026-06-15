"""
Generate synthetic antipsychotic prescription data for testing the pipeline.

Creates data/antipsychotic_prescriptions.csv with 60 simulated patients, each having
1–3 sequential medication episodes spanning several years.  The file uses
the same column names and format as real prescription data.

Usage
-----
    python generate_sample_data.py
"""

import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# Drug name → list of typical doses (mg).
# Keys match the normalised drug_name values in antipsychotics_lookup.json.
DRUGS = {
    # Atypical (second-generation)
    'amisulpride':      [200, 400, 600, 800],
    'aripiprazole':     [10, 15, 20, 30],
    'asenapine':        [5, 10],
    'cariprazine':      [1.5, 3, 4.5, 6],
    'clozapine':        [100, 200, 300, 400],
    'lurasidone':       [37, 74, 111, 148],
    'olanzapine':       [5, 10, 15, 20],
    'paliperidone':     [3, 6, 9, 12],
    'quetiapine':       [100, 200, 300, 400],
    'risperidone':      [2, 4, 6],
    'sulpiride':        [200, 400, 800],
    # Typical (first-generation)
    'benperidol':       [0.25, 0.5, 1, 1.5],
    'chlorpromazine':   [50, 100, 200, 300],
    'flupentixol':      [3, 6, 9],
    'fluphenazine':     [12.5, 25, 50, 100],
    'haloperidol':      [5, 10, 15, 20],
    'levomepromazine':  [25, 50, 100],
    'pericyazine':      [15, 25, 75],
    'pimozide':         [2, 4, 8],
    'prochlorperazine': [5, 10, 15],
    'promazine':        [25, 50, 100],
    'trifluoperazine':  [5, 10, 15],
    'zuclopenthixol':   [20, 25, 50],
}

N_PATIENTS = 60
records = []

for pid in range(1001, 1001 + N_PATIENTS):
    # Each patient has 1–3 sequential drug episodes
    n_drugs = random.choices([1, 2, 3], weights=[25, 50, 25])[0]
    drugs = random.sample(list(DRUGS.keys()), n_drugs)

    current_date = (
        pd.Timestamp('2012-01-01') + pd.Timedelta(days=random.randint(0, 365 * 4))
    )

    for drug in drugs:
        dose = random.choice(DRUGS[drug])
        ep_days = random.randint(180, 540)   # episode length: 6–18 months

        # Roughly monthly prescriptions with ±1-week jitter
        rx_date = current_date
        while rx_date < current_date + pd.Timedelta(days=ep_days):
            rx_dose = dose if random.random() > 0.15 else random.choice(DRUGS[drug])
            # 10% of records come from a supplementary source without structured dose fields
            source = 'supplementary' if random.random() < 0.1 else 'primary'
            records.append({
                'patient_id':  pid,
                'drug_name':   drug,
                'date':        rx_date.strftime('%Y-%m-%d'),
                'dose':        f'{rx_dose}mg',
                'source_table': source,
            })
            rx_date += pd.Timedelta(days=28 + random.randint(-7, 7))

        # Short gap between episodes (2–8 weeks)
        current_date += pd.Timedelta(days=ep_days + random.randint(14, 56))

df = (
    pd.DataFrame(records)
    .sort_values(['patient_id', 'date'])
    .reset_index(drop=True)
)

os.makedirs('data', exist_ok=True)
out = 'data/antipsychotic_prescriptions.csv'
df.to_csv(out, index=False)

print(f"Saved {len(df):,} prescription records for {df['patient_id'].nunique()} patients → {out}")
print("\nDrug distribution:")
print(df['drug_name'].value_counts().to_string())
