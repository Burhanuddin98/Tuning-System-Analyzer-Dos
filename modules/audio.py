# modules/audio.py
from __future__ import annotations
import io, base64
import numpy as np
import soundfile as sf
from typing import Tuple, Optional

def decode_for_readout(raw: bytes) -> Tuple[Optional[np.ndarray], Optional[int]]:
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, sr
    except Exception:
        return None, None

def to_wav_b64(raw: bytes) -> str:
    y, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode('ascii')

