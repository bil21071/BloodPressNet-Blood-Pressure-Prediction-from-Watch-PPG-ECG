import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from scipy.stats import skew, kurtosis


def find_first_maximum(x):
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            return i
    return 0


def find_second_maximum(x):
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    return int(peaks[1]) if peaks.size >= 2 else 0


def find_first_minimum(x):
    for i in range(1, len(x) - 1):
        if x[i] < x[i - 1] and x[i] < x[i + 1]:
            return i
    return 0


def minmax_norm(x):
    mn, mx = np.min(x), np.max(x)
    denom = (mx - mn) if (mx - mn) > 0 else 1.0

    return (x - mn) / denom


def norm_idx(i, T):
    # convert absolute index -> time-normalized [0,1]
    return i / max(T - 1, 1)


def fiducials(raw_beat, fs):
    beat = minmax_norm(raw_beat)

    first = savgol_filter(beat, window_length=19, polyorder=7, delta=1 / fs, deriv=1)
    second = savgol_filter(beat, window_length=19, polyorder=7, delta=1 / fs, deriv=2)
    third = savgol_filter(beat, window_length=19, polyorder=7, delta=1 / fs, deriv=3)

    '''
    Fiducials
    # TODO: MULTIPLE FALLBACKS
    '''
    T = len(beat)
    s = np.argmax(beat)
    # print(f"S; the maximum of x: {s}")

    ms = np.argmax(first)
    # print(f"ms; the maximum of x': {ms}")
    e_ = f"success"
    try:
        e = int(find_second_maximum(second[ms:int(.6 * T)]) + ms)
        # print(f"e; the second maximum of x'' between ms and .6T: {e}")
        # TODO: Evaluate this fallback. Reminder: separate derivations!
        # if e == ms:
        #     e = int(find_first_maximum(second[ms:int(.6*T)]) + ms)
        #     e_ = f"fallback"
        # print(f"e; the first maximum of x'' between ms and .6T: {e}")
    except (TypeError, IndexError):
        return None, None, None, None, None
        # print(f"e; None: {e}")

    dic = e
    # print(f"dic; same as e")
    # TODO: FALLBACK
    dia_ = f"success"
    dia = int(find_first_maximum(beat[dic:int(.8 * T)]) + dic)
    # print(f"dia; first maximum of x between dic and .8T")
    if dia == dic:
        dia_ = f"fallback"
        dia = int(find_first_maximum(second[e:int(.8 * T)]) + e)

    if ms == 0:
        return None, None, None, None, None
    a = np.argmax(second[:ms])
    # print(f"a; global maximum of x'' before ms")

    b = int(find_first_minimum(second[a:]) + a)

    if b >= e:
        return None, None, None, None, None
    c_ = f"success"
    c = int(np.argmax(second[b:e]) + b)
    # print(f"c; global maximum of x'' after b and before e")
    if (c == b) or (c == e):
        c_ = f"first fallback"
        c = int(find_first_maximum(first[e:]) + e)
        if c == e:
            c_ = f"second fallback"
            c = find_first_minimum(third[e:] + e)
        # print(f"c; first maximum of x' after e")

    d_ = f"success"
    d = int(np.argmin(second[c:e]) + c)
    # print(f"d; global minimum of x'' after c and before e")
    if (d == c) or (d == e):
        d_ = f"fallback"
        d = c
        # print(f"d; same as c")

    # f = int(find_first_minimum(second[e:int(.8*T)]) + e)
    # print(f"e: {e}, first minimum: {find_first_minimum(second[e:int(.8*T)])}")
    f = int(find_first_minimum(second[e:int(.8 * T)]) + e)
    # print(f"f; first minimum of x'' after e")

    if not (s < dic < dia) or not (a < b < c < d < e < f):
        return None, None, None, None, None

    # fig, axes = plt.subplots(4, 1)
    # axes[0].plot(beat, label="PPG Beat")
    # axes[0].scatter(s, beat[s], label="S (max)")
    # axes[0].scatter(dic, beat[dic], label="N (same as e)")
    # axes[0].scatter(dia, beat[dia], label=f"D: {dia_}")

    # axes[1].plot(first, label="VPG")
    # axes[1].scatter(ms, first[ms], label="ms (max)")
    # axes[1].scatter(dic, first[dic], label="N (same as e)")
    # axes[1].scatter(dia, first[dia], label=f"D: {dia_}")

    # axes[2].plot(second, label='APG')
    # axes[2].scatter(a, second[a], label="a (max)")
    # axes[2].scatter(b, second[b], label="b (first min)")
    # axes[2].scatter(c, second[c], label=f"c: {c_}")
    # axes[2].scatter(d, second[d], label="d (min)")
    # axes[2].scatter(e, second[e], label=f"e: {e_}")
    # axes[2].scatter(f, second[f], label="f (first min)")

    # axes[3].plot(third, label="Third")

    # for ax in axes:
    #     ax.legend()
    # plt.show()

    fids = {
        's': s,
        'dia': dia,
        'n': dic,
        'ms': ms,
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
        'f': f,
    }

    # Amplitude normalization only for the x (ppg) signal not for its derivatives!
    fids_amp = {
        's': beat[s],
        'dia': beat[dia],
        'n': beat[dic],
        'ms': first[ms],
        'a': second[a],
        'b': second[b],
        'c': second[c],
        'd': second[d],
        'e': second[e],
        'f': second[f],
    }

    fids_idx_norm = {k: norm_idx(v, T) for k, v in fids.items()}

    results = {
        'idx': fids,  # absolute indices
        't': fids_idx_norm,  # 0..1 time-normalized locations
        'amp': fids_amp,  # 0..1 amplitudes (for x); derivatives are as-is
        'T': T,
    }

    return results, beat, first, second, third


def slope(x1, y1, x2, y2):
    s = (y2 - y1) / (x2 - x1)
    return s


def width(beat, fraction=0.25):
    thr = fraction
    above = beat >= thr

    crossings = np.where(np.diff(above.astype(int)) != 0)[0]

    left, right = crossings[0], crossings[-1]
    width_samples = right - left

    return width_samples / (len(beat) - 1)


def extraction(fiducials, beat, first, second, third):
    '''
    IMPORTANT:
    results[i]['t01'][fid] → each fiducial’s time-normalized position ∈ [0,1].
    results[i]['amp01']['s'|'d'|'n'] → amplitude-normalized (0–1) values for the original signal at those points.
    Absolute indices are still there if you need them.
    '''
    t_sys = fiducials['t']['n']
    t_dia = 1 - fiducials['t']['n']
    try:
        t_ratio = t_sys / t_dia
    except ZeroDivisionError:
        t_ratio = 0

    a1 = np.trapz(beat[:fiducials['idx']['n']], dx=1 / fiducials['T'])
    a2 = np.trapz(beat[fiducials['idx']['n']:], dx=1 / fiducials['T'])

    try:
        ipa = a2 / a1
    except ZeroDivisionError:
        ipa = 0

    features = {
        'notch_amp': fiducials['amp']['n'],
        'reflective_idx': fiducials['amp']['dia'],
        'delta_T': fiducials['t']['dia'] - fiducials['t']['s'],
        'crest_time': fiducials['t']['s'],
        'T_sys': t_sys,
        'T_dia': t_dia,
        'T_ratio': t_ratio,
        'stt': slope(0, 0, fiducials['t']['s'], fiducials['amp']['s']),
        # TODO: Extend
        # Stress-Induced Vascular Response Index (sVRI)
        'A1': a1,
        'A2': a2,
        'inflection_point_area': ipa,
        'width_25': width(beat, .25),
        'width_50': width(beat, .5),
        # TODO: Extend
        # Pressure index (PI)
        # Normalised harmonic area (NHA)
        # Inflection and harmonic area ratio (IHAR)
        # skewness
        'skew': skew(beat),
        'kurtosis': kurtosis(beat),
        # kurtosis

        ### VPG Morph
        'sys_mu': np.mean(first[:fiducials['idx']['n']]),
        'sys_sigma': np.var(first[:fiducials['idx']['n']]),
        'dia_mu': np.mean(first[fiducials['idx']['n']:]),
        'dia_sigma': np.var(first[fiducials['idx']['n']:]),

        ### APG Morph

    }

    return features