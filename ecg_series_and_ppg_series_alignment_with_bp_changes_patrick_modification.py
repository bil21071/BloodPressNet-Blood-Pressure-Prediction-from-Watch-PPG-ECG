#!/usr/bin/env python3
"""
Extract aligned PPG (Pleth) and ECG (Lead II) windows around BP-change events,
compute PPG features per window, compute mean RR interval from ECG,
and save everything into a single CSV.
"""

import os
import csv
import wfdb
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Optional, Tuple, List, Dict

import features  # your custom PPG features module
import onset as onsets  # your custom module

# ---------------- CONFIG ----------------
PATH = '/Users/bilal_mahnoor/Pictures/physionet.org/files/mimic4wdb/0.1.0/'
OUT_CSV = '/Users/bilal_mahnoor/Documents/bloodpressurecodesetup/PATRICK_MODIFIED10011.csv'
OUT_WIN_DIR = '//window_patrick10011'

STOP_AT = "85583557"  # inclusive; set None to process all
PRE_S = 10.0  # seconds before event
POST_S = 30.0  # seconds after event

os.makedirs(OUT_WIN_DIR, exist_ok=True)

BASE_FIELDNAMES = [
    "record_path",
    "event_time_iso",
    "SBP",
    "DBP",
    "window_start_iso",
    "window_end_iso",
    "ppg_start_idx",
    "ppg_end_idx",
    "ecg_start_idx",
    "ecg_end_idx",
    "window_n_samples_ppg",
    "window_n_samples_ecg",
    "fs",
    "ppg_npy_path",
    "ecg_npy_path",
    "n_rpeaks",
    "mean_rr_ms"
]

# ---------------- FUNCTIONS ----------------
def read_record(path: str) -> Optional[Tuple[pd.Series, pd.Series, pd.DataFrame, float, pd.Timestamp]]:
    """Read record and return aligned PPG, Lead II, BP, fs, base_dt."""
    try:
        header = wfdb.rdheader(path, rd_segments=True)
    except Exception as e:
        print(f"[skip] header error {path}: {e}")
        return None

    numeric_path = f"{path}n.csv.gz"
    if not os.path.exists(numeric_path):
        return None
    try:
        numeric = pd.read_csv(numeric_path)
    except Exception as e:
        print(f"[skip] numeric read error {numeric_path}: {e}")
        return None

    try:
        base_dt = pd.to_datetime(header.base_datetime)
    except Exception as e:
        print(f"[skip] invalid base_datetime {path}: {e}")
        return None

    if not hasattr(header, "counter_freq") or header.counter_freq is None:
        return None

    numeric['dt'] = base_dt + pd.to_timedelta(numeric['time'] / header.counter_freq, unit='s')
    numeric = numeric.set_index('dt').sort_index()

    if ('NBPs [mmHg]' in numeric) and ('NBPd [mmHg]' in numeric):
        bp = numeric[['NBPs [mmHg]', 'NBPd [mmHg]']].dropna()
        if len(bp) > 2:
            if ('Pleth' in header.sig_name) and ('II' in header.sig_name):
                pleth_idx = header.sig_name.index('Pleth')
                lead2_idx = header.sig_name.index('II')
                fs = getattr(header, "fs", None)
                if fs is None:
                    return None
                try:
                    record = wfdb.rdrecord(path, channels=[pleth_idx, lead2_idx])
                except Exception as e:
                    print(f"[skip] rdrecord failed {path}: {e}")
                    return None

                try:
                    ppg = record.p_signal[:, 0].flatten()
                    lead2 = record.p_signal[:, 1].flatten()
                except Exception as e:
                    return None

                n_samples = ppg.shape[0]
                wave_index = pd.to_datetime(base_dt) + pd.to_timedelta(np.arange(n_samples) / fs, unit='s')
                return pd.Series(ppg, index=wave_index), pd.Series(lead2, index=wave_index), bp, fs, base_dt
    return None

def bp_changes(bp_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where SBP or DBP changed vs previous row."""
    bp = bp_df.copy()
    bp['diff_sys'] = bp['NBPs [mmHg]'].diff()
    bp['diff_dia'] = bp['NBPd [mmHg]'].diff()
    filtered = bp[(bp['diff_sys'] != 0) | (bp['diff_dia'] != 0)].drop(['diff_sys','diff_dia'], axis=1)
    return filtered

def extract_windows_pair(ppg_series: pd.Series,
                         ecg_series: pd.Series,
                         filtered_bp: pd.DataFrame,
                         base_dt: pd.Timestamp,
                         fs: float,
                         pre_s: float = PRE_S,
                         post_s: float = POST_S) -> List[Dict]:
    """Extract PPG+ECG windows around BP-change events."""
    windows = []
    n_samples = len(ppg_series)
    for t in filtered_bp.index:
        sbp = float(filtered_bp.loc[t, 'NBPs [mmHg]'])
        dbp = float(filtered_bp.loc[t, 'NBPd [mmHg]'])
        event_offset_s = (t - base_dt).total_seconds()
        start_idx = int(round((event_offset_s - pre_s) * fs))
        end_idx = int(round((event_offset_s + post_s) * fs))
        clip_start = max(0, start_idx)
        clip_end = min(n_samples - 1, end_idx)
        if clip_end < clip_start:
            continue

        ppg_vals = ppg_series.iloc[clip_start:clip_end + 1].values
        ecg_vals = ecg_series.iloc[clip_start:clip_end + 1].values
        if len(ppg_vals) == 0 or np.all(np.isnan(ppg_vals)):
            continue

        windows.append({
            "event_time": t,
            "sbp": sbp,
            "dbp": dbp,
            "ppg_values": ppg_vals,
            "ecg_values": ecg_vals,
            "ppg_start_idx": int(clip_start),
            "ppg_end_idx": int(clip_end),
            "ecg_start_idx": int(clip_start),
            "ecg_end_idx": int(clip_end),
            "window_start_time": ppg_series.index[clip_start],
            "window_end_time": ppg_series.index[clip_end],
            "fs": float(fs)
        })
    return windows

# ---------------- MAIN SCRIPT ----------------
def main():
    fieldnames = BASE_FIELDNAMES.copy()
    features_rows = []
    window_id_counter = 0

    header_exists = os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0
    with open(OUT_CSV, "a", newline="") as fcsv:
        writer = None

        stop_reached = False
        records_idx = os.path.join(PATH, "RECORDS")
        if not os.path.exists(records_idx):
            raise FileNotFoundError(f"RECORDS file not found at {records_idx}")

        with open(records_idx, newline='') as fh:
            for pat_r in csv.reader(fh):
                if stop_reached: break
                if not pat_r: continue
                pat_r_str = pat_r.pop()
                records_file = os.path.join(PATH, pat_r_str + "RECORDS")
                if not os.path.exists(records_file): continue

                with open(records_file, newline='') as fh2:
                    for pat in csv.reader(fh2):
                        if stop_reached: break
                        if not pat: continue
                        pat_str = pat.pop()
                        abs_path = os.path.join(PATH, pat_r_str + pat_str)
                        print("Processing:", pat_str)

                        res = read_record(abs_path)
                        if res is None: continue
                        ppg_series, lead2_series, bp_df, fs, base_dt = res
                        filtered = bp_changes(bp_df)
                        if filtered.empty: continue

                        windows = extract_windows_pair(ppg_series, lead2_series, filtered, base_dt, fs, PRE_S, POST_S)

                        for w in windows:
                            # -------- PPG features --------
                            try:
                                ppg_clean = nk.ppg_clean(w['ppg_values'], sampling_rate=fs)
                            except:
                                continue

                            ssf = onsets.compute_ssf(ppg_clean, fs=fs)
                            bssf_boxes = onsets.find_bssf_boxes(ssf)
                            beat_onsets, _, _ = onsets.detect_beat_onsets(ssf, bssf_boxes, fs=fs)

                            raw_beats = []
                            for i in range(len(beat_onsets)-1):
                                start = beat_onsets[i]
                                end = beat_onsets[i+1]
                                if end > start:
                                    raw_beats.append(ppg_clean[start:end])

                            features_all_beats = []
                            for raw_b in raw_beats:
                                if len(raw_b) < 19: continue
                                fids, new_beat, first, second, third = features.fiducials(raw_b, fs=fs)
                                if fids is None: continue
                                features_dict = features.extraction(fids, new_beat, first, second, third)
                                features_all_beats.append(features_dict)

                            if len(features_all_beats) == 0:
                                continue

                            df_feat = pd.DataFrame(features_all_beats)
                            mean_features = df_feat.apply(np.nanmean)
                            feat_dict = {f"feat_{k}": (None if pd.isna(v) else float(v)) for k,v in mean_features.items()}

                            # Update fieldnames dynamically
                            for k in feat_dict.keys():
                                if k not in fieldnames:
                                    fieldnames.append(k)
                                    if writer:  # update writer if already exists
                                        writer.fieldnames = fieldnames

                            # -------- ECG: mean RR interval --------
                            ecg_window = w['ecg_values']
                            rec_start = w['window_start_time']
                            try:
                                ecg_clean = nk.ecg_clean(ecg_window, sampling_rate=fs)
                                _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method="hamilton")
                                r_local = np.asarray(info["ECG_R_Peaks"], dtype=int)
                                if len(r_local) < 2:
                                    n_rpeaks = 0
                                    mean_rr = None
                                else:
                                    rr_ms = np.diff(r_local * (1000.0 / fs))
                                    n_rpeaks = len(r_local)
                                    mean_rr = float(np.nanmean(rr_ms))
                            except:
                                n_rpeaks = 0
                                mean_rr = None

                            # -------- Save window arrays + metadata --------
                            ev_str = w['event_time'].strftime("%Y%m%dT%H%M%S.%f")
                            safe_rec = pat_str.replace("/", "_").replace("\\", "_")
                            ppg_fname = f"{safe_rec}__{ev_str}__ppg_s{w['ppg_start_idx']}_e{w['ppg_end_idx']}.npy"
                            ecg_fname = f"{safe_rec}__{ev_str}__ecg_s{w['ecg_start_idx']}_e{w['ecg_end_idx']}.npy"

                            np.save(os.path.join(OUT_WIN_DIR, ppg_fname), w['ppg_values'])
                            np.save(os.path.join(OUT_WIN_DIR, ecg_fname), w['ecg_values'])

                            meta_row = {
                                "record_path": abs_path,
                                "event_time_iso": w['event_time'].isoformat(),
                                "SBP": w['sbp'],
                                "DBP": w['dbp'],
                                "window_start_iso": w['window_start_time'].isoformat(),
                                "window_end_iso": w['window_end_time'].isoformat(),
                                "ppg_start_idx": w['ppg_start_idx'],
                                "ppg_end_idx": w['ppg_end_idx'],
                                "ecg_start_idx": w['ecg_start_idx'],
                                "ecg_end_idx": w['ecg_end_idx'],
                                "window_n_samples_ppg": int(len(w['ppg_values'])),
                                "window_n_samples_ecg": int(len(w['ecg_values'])),
                                "fs": w['fs'],
                                "ppg_npy_path": os.path.join(OUT_WIN_DIR, ppg_fname),
                                "ecg_npy_path": os.path.join(OUT_WIN_DIR, ecg_fname),
                                "n_rpeaks": n_rpeaks,
                                "mean_rr_ms": mean_rr
                            }

                            # Initialize writer if not already
                            if writer is None:
                                writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
                                if not header_exists:
                                    writer.writeheader()
                                    header_exists = True

                            writer.writerow({**meta_row, **feat_dict})
                            features_rows.append({**meta_row, **feat_dict})
                            window_id_counter += 1

                        if STOP_AT is not None and (pat_str == STOP_AT or abs_path.endswith(f"/{STOP_AT}") or f"/{STOP_AT}/" in abs_path):
                            stop_reached = True
                            break

    print("Done. All windows saved to single CSV:", OUT_CSV)
    print("PPG/ECG windows saved to:", OUT_WIN_DIR)


if __name__ == "__main__":
    main()
