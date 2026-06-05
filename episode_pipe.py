"""
Detect antipsychotic medication episodes from structured prescription records.

Uses PELT (Pruned Exact Linear Time) change-point detection to segment each
patient's prescription history into discrete medication episodes, then
identifies concurrent / transitional polypharmacy periods.

Input : data/antipsychotic_prescriptions.csv  (one row per prescription)
Output: results/episodes.csv, results/episodes_sequences.pkl,
        results/episodes_polypharmacy.pkl
"""
# pip install ruptures tqdm pandas numpy
import multiprocessing
import os
import pickle
import ruptures as rpt
import pandas as pd
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Load and preprocess antipsychotic prescription data
# ---------------------------------------------------------------------------
aps = pd.read_csv('data/antipsychotic_prescriptions.csv', low_memory=False)

aps.columns = aps.columns.str.strip()
aps['date'] = pd.to_datetime(aps['date'], errors='coerce')
aps['drug_name'] = aps['drug_name'].str.lower().str.strip()
print(aps['drug_name'].value_counts())

# Extract numeric dose value; keep rows with a parseable dose or from a
# supplementary source whose rows lack a structured dose field.
aps['dose_num'] = aps['dose'].str.lower().str.extract(r'(\d+\.?\d* *)[mng]')
aps = aps[aps['dose_num'].notna() | aps['source_table'].str.startswith('supplementary')]

aps['drug_name_raw'] = aps['drug_name']

# Retain only patients with more than 5 records (enough signal for change-point detection)
pc = aps.groupby('patient_id').size().reset_index(name='count')
aps = aps[aps['patient_id'].isin(pc[pc['count'] > 5]['patient_id'])]
aps = aps.sort_values(['patient_id', 'date']).reset_index(drop=True)

print(aps.shape, aps['patient_id'].nunique())


# ---------------------------------------------------------------------------
# Polypharmacy detection
# ---------------------------------------------------------------------------
def detect_polypharmacy_periods(results, dg, exclude_augmentation=None):
    """
    Detect polypharmacy periods around change points.

    Args:
        results: List of medication episodes, each entry:
                 [patient_id, order, medication, date, end_date, dose].
        dg: DataFrame with columns ['drug_name', 'date'] for the
            current patient only.
        exclude_augmentation: Medications to exclude from polypharmacy
            detection (e.g. ['aripiprazole']).  Defaults to an empty list.

    Returns:
        List of dicts describing detected polypharmacy periods.
    """
    if exclude_augmentation is None:
        exclude_augmentation = []

    polypharmacy_periods = []

    for i, episode in enumerate(results):
        patient_id, order, med, start, end, dose = episode

        # Normalise to pandas Timestamp
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        # ------------------------------------------------------------------
        # 1. Transition overlap: current med still mentioned after next starts
        # ------------------------------------------------------------------
        if i + 1 < len(results):
            next_episode = results[i + 1]
            next_med = next_episode[2]
            next_start = pd.Timestamp(next_episode[3])

            # Not a real transition if the medication name is unchanged
            if med.lower() == next_med.lower():
                continue

            # Ignore augmentation / side-effect management drugs
            if any(aug in next_med.lower() for aug in exclude_augmentation):
                continue

            transition_window = 14  # days either side of the change point
            window_start = end - pd.Timedelta(days=transition_window)
            window_end = next_start + pd.Timedelta(days=transition_window)

            current_in_window = dg[
                (dg['drug_name'] == med) &
                (dg['date'] >= window_start) &
                (dg['date'] <= window_end)
            ]
            next_in_window = dg[
                (dg['drug_name'] == next_med) &
                (dg['date'] >= window_start) &
                (dg['date'] <= window_end)
            ]

            if len(current_in_window) >= 2 and len(next_in_window) >= 2:
                # Current med must be mentioned *after* the next med first appears
                overlap_mentions = current_in_window[
                    current_in_window['date'] >= next_in_window['date'].min()
                ]
                if len(overlap_mentions) >= 2:
                    polypharmacy_periods.append({
                        'patient_id': patient_id,
                        'type': 'transition_overlap',
                        'primary_med': med,
                        'secondary_med': next_med,
                        'period_start': window_start,
                        'period_end': window_end,
                        'overlap_days': (
                            overlap_mentions['date'].max() -
                            next_in_window['date'].min()
                        ).days,
                    })

        # ------------------------------------------------------------------
        # 2. Concurrent therapy: another drug appears substantially during
        #    the same episode window
        # ------------------------------------------------------------------
        episode_records = dg[
            (dg['date'] >= start) &
            (dg['date'] <= end)
        ]
        concurrent_counts = episode_records['drug_name'].value_counts()

        # Require at least 3 mentions, or roughly 1 per month — computed once per episode
        episode_duration_days = (end - start).days
        min_mentions = max(3, episode_duration_days // 30)

        for concurrent_med, count in concurrent_counts.items():
            if concurrent_med.lower() == med.lower():
                continue
            if any(aug in concurrent_med.lower() for aug in exclude_augmentation):
                continue
            if count >= min_mentions:
                polypharmacy_periods.append({
                    'patient_id': patient_id,
                    'type': 'concurrent_therapy',
                    'primary_med': med,
                    'secondary_med': concurrent_med,
                    'period_start': start,
                    'period_end': end,
                    'secondary_mentions': count,
                })

    return polypharmacy_periods


# ---------------------------------------------------------------------------
# Episode extraction via change-point detection
# ---------------------------------------------------------------------------
def medication_timelines_improved(patient_id, penalty=None, min_size=2, kernel='rbf'):
    """
    Detect medication episodes for a single patient using PELT change-point
    detection, then identify polypharmacy periods around each change point.

    dat includes all rows so the full prescription history feeds the
    polypharmacy checks in dg.  Dose averaging uses dose_dat, which keeps
    only rows with a reliable numeric dose.

    Args:
        patient_id: Patient identifier.
        penalty: PELT penalty value (higher → fewer breakpoints).  If None
                 (default), an adaptive BIC-style penalty log(n) is computed
                 per patient, scaling with signal length only.
        min_size: Minimum segment length for PELT.
        kernel: PELT kernel ('rbf', 'l2', etc.).

    Returns:
        results: List of episodes [patient_id, order, medication, start, end, dose].
        med_seq: Ordered list of dominant medications across episodes.
        polypharmacy_periods: List of detected polypharmacy period dicts.
    """
    results = []
    med_seq = []

    # All rows for this patient — used for polypharmacy timeline
    dat = aps[aps['patient_id'] == patient_id]

    # Per-patient timeline for polypharmacy lookups.
    # Order is guaranteed by the global sort on aps after data loading.
    dg = dat[['drug_name', 'date']].copy()

    # Assign integer codes ordered by first appearance date so the signal
    # reflects the temporal order in which drugs were introduced
    first_appearances = dg.drop_duplicates(subset=['drug_name'], keep='first')
    dg['drug_name'] = (
        dg['drug_name']
        .astype('category')
        .cat.set_categories(first_appearances['drug_name'], ordered=True)
    )
    dg['drug_name_id'] = dg['drug_name'].cat.codes

    signal = np.array(dg['drug_name_id'])
    signal2 = np.array(dg['drug_name'])
    dates = np.array(dg['date'])

    # Adaptive jump: 0.1% of signal length, minimum 1
    jump = max(1, int(len(signal) * 0.001))
    algo = rpt.Pelt(model=kernel, jump=jump, min_size=min_size).fit(signal)

    # BIC-style adaptive penalty: scales with signal length so longer records
    # require stronger evidence to add a breakpoint.  The RBF kernel cost is
    # already normalised, so var(signal) is not included (it would over-penalise
    # patients with many different medications).
    if penalty is None:
        pen = np.log(len(signal))
    else:
        pen = penalty
    # predict() returns 1-based end indices; prepend 1 to serve as segment start
    breakpoints = [1] + algo.predict(pen=pen)

    # Dose averaging uses only rows with a parseable numeric dose (excludes supplementary source)
    dose_dat = dat[~dat['source_table'].str.startswith('supplementary')].copy()
    dose_dat['dose_num'] = pd.to_numeric(dose_dat['dose_num'], errors='coerce')

    for i in range(len(breakpoints) - 1):
        # Convert 1-based breakpoints to 0-based array indices
        seg_start_idx = breakpoints[i] - 1
        seg_end_idx = breakpoints[i + 1] - 1

        date = pd.Timestamp(dates[seg_start_idx])
        end_date = pd.Timestamp(dates[seg_end_idx])

        # Dominant medication in this segment (inclusive of boundary point)
        episode_meds = signal2[seg_start_idx:seg_end_idx + 1]
        cmed = pd.value_counts(episode_meds).index[0]

        # ------------------------------------------------------------------
        # Boundary date correction
        #
        # seg_end_idx is a shared boundary: its date is used as end_date but
        # the prescription at that index may belong to the next drug.
        # We shift:
        #   date     → first mention of cmed in this segment  (Case 1)
        #   end_date → last  mention of cmed in this segment  (Case 2)
        # but ONLY when the gap contains no other medication records.
        # The slice must be inclusive of seg_end_idx so that a boundary
        # prescription belonging to cmed is not mistakenly treated as a gap.
        # ------------------------------------------------------------------
        cmed_mask = episode_meds == cmed
        if cmed_mask.any():
            seg_dates = dates[seg_start_idx:seg_end_idx + 1]
            cmed_seg_dates = seg_dates[cmed_mask]
            first_cmed_date = pd.Timestamp(cmed_seg_dates[0])
            last_cmed_date  = pd.Timestamp(cmed_seg_dates[-1])

            # Case 1: breakpoint landed on previous med's last mention —
            # push date forward to cmed's first mention if no other
            # medication appears in the segment during the gap.
            if first_cmed_date > date:
                has_meds_before = np.any(
                    (seg_dates > date) & (seg_dates < first_cmed_date)
                )
                if not has_meds_before:
                    date = first_cmed_date

            # Case 2: breakpoint landed on next med's first mention —
            # pull end_date backward to cmed's last mention if no other
            # medication appears in the segment during the gap.
            if last_cmed_date < end_date:
                has_meds_after = np.any(
                    (seg_dates > last_cmed_date) & (seg_dates < end_date)
                )
                if not has_meds_after:
                    end_date = last_cmed_date

        # Average dose: inclusive on both ends
        dose = dose_dat[
            (dose_dat['drug_name'] == cmed) &
            (dose_dat['date'] >= date) &
            (dose_dat['date'] <= end_date)
        ]['dose_num'].mean()

        results.append([patient_id, i, cmed, date, end_date, dose])
        med_seq.append(cmed)

    polypharmacy_periods = detect_polypharmacy_periods(results, dg)

    return results, med_seq, polypharmacy_periods


# TODO: detect medication restarts — a threshold could flag them, but it is
#       ambiguous whether a gap represents a true restart or missing data.


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------
def _worker_init(aps_df):
    """Initialise each worker process with a shared copy of the DataFrame."""
    global aps
    aps = aps_df


def _process_patient(args):
    """
    Top-level wrapper required for pickling with multiprocessing.Pool.

    Returns (patient_id, results, med_seq, polypharmacy_periods) on success, or
    (patient_id, None, error_string, None) on failure so the main process can log the error.
    """
    patient_id, min_size = args
    try:
        r, med_seq, poly = medication_timelines_improved(patient_id, min_size=min_size)
        return patient_id, r, med_seq, poly
    except Exception as e:
        return patient_id, None, str(e), None


# ---------------------------------------------------------------------------
# Run pipeline over all patients (parallel)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    patient_ids = aps['patient_id'].unique().tolist()
    args_list = [(patient_id, 2) for patient_id in patient_ids]

    n_workers = max(1, os.cpu_count() - 1)  # leave one core free for the OS
    print(f"Processing {len(patient_ids)} patients using {n_workers} workers …")

    os.makedirs('results', exist_ok=True)

    res, med_seqs, polys = [], {}, {}

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(aps,),
    ) as pool:
        for patient_id, r, med_seq, poly in tqdm(
            pool.imap_unordered(_process_patient, args_list, chunksize=10),
            total=len(patient_ids),
        ):
            if r is None:
                print(f"Error processing patient {patient_id}: {med_seq}")  # med_seq carries the error msg
            elif r:
                res.extend(r)
                med_seqs[patient_id] = med_seq
                polys[patient_id] = poly

    timelines = pd.DataFrame(res, columns=['patient_id', 'order', 'medication', 'date', 'end_date', 'dose'])

    timelines.to_csv('results/episodes.csv', index=False)
    pickle.dump(med_seqs, open('results/episodes_sequences.pkl', 'wb'))
    pickle.dump(polys, open('results/episodes_polypharmacy.pkl', 'wb'))
    print("Done.")
