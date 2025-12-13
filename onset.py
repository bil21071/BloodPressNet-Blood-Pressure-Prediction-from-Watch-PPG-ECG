import numpy as np
import matplotlib.pyplot as plt

# --- SSF
def compute_ssf(signal, fs, window_size=170):
    w = int((window_size / 1000) * fs)
    signal = np.asarray(signal)
    dx = np.diff(signal, prepend=signal[0])
    u = np.where(dx > 0, dx, 0)
    ssf = np.full_like(signal, np.nan, dtype=np.float64)
    for i in range(w, len(signal)):
        ssf[i] = np.sum(u[i - w:i + 1])
    return ssf

# --- BSSF boxes
def find_bssf_boxes(ssf_signal):
    ssf = np.nan_to_num(ssf_signal)
    starts = np.where((ssf[:-1] == 0) & (ssf[1:] > 0))[0] + 1
    ends = np.where((ssf[:-1] > 0) & (ssf[1:] == 0))[0] + 1
    boxes = []
    for s in starts:
        e = ends[ends > s]
        if len(e) == 0:
            continue
        boxes.append((s, e[0]))

    # plt.plot(ssf, label="SSF Signal")

    # for s, e in boxes:
    #     plt.axvspan(s, e, color="red", alpha=0.3)  # shaded box

    # plt.legend()
    # plt.show()

    return boxes

# --- Beat onset detection using box start (uBSSF) + adaptive threshold
def detect_beat_onsets(ssf_signal, boxes, fs):
    # Use all box start points (uBSSF) as onsets
    onsets = [start for start, _ in boxes]

    # Skip the first one (may be mid-beat)
    onsets = onsets[1:]

    return np.array(onsets, dtype=int), np.array(onsets, dtype=int), None

# --- Detect peaks between beat onsets
def detect_peaks(ppg_signal, onsets):
    peaks = []
    for i in range(len(onsets) - 1):
        start = onsets[i]
        end = onsets[i + 1]
        if end <= start:
            continue
        peak_idx = np.argmax(ppg_signal[start:end]) + start
        peaks.append(peak_idx)
    return np.array(peaks, dtype=int)


