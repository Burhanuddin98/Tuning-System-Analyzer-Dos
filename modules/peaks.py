# modules/peaks.py
from __future__ import annotations
import numpy as np
from scipy.signal import stft, find_peaks

def amplitude_to_db(x, ref=1.0, amin=1e-12):
    x = np.maximum(x, amin)
    return 20.0 * np.log10(x / ref)

def extract_global_peaks(y, sr, fmin: float, fmax: float, peak_db: float):
    """Lightweight global peak pool for tuning readout."""
    nperseg = int(sr * 0.064)
    hop = int(sr * 0.024)
    noverlap = max(0, nperseg - hop)
    f, t, Z = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    S = np.abs(Z)
    mag_db = amplitude_to_db(S, ref=np.max(S)+1e-12)
    mask = (f >= fmin) & (f <= fmax)
    fbin = f[mask]
    all_f, all_m = [], []
    for ti in range(mag_db.shape[1]):
        spec = mag_db[mask, ti]
        if spec.size < 5: continue
        pk, props = find_peaks(spec, height=peak_db)
        if pk.size == 0: continue
        h = props["peak_heights"]
        sel = np.argsort(h)[-12:][::-1]
        all_f.append(fbin[pk[sel]]); all_m.append(h[sel])
    if all_f:
        return np.concatenate(all_f), np.concatenate(all_m)
    return np.array([]), np.array([])


