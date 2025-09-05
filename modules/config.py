# modules/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass
class SpiralUI:
    turns: int
    bins_per_turn: int
    spokes: int
    line_width: int
    max_segments: int
    def to_dict(self): return asdict(self)

@dataclass
class RippleUI:
    wiggle_gain: float
    ripple_target: float
    ripple_max: float
    ripple_gamma: float
    ripple_temporal_cutoff: float
    smooth_bins: int
    def to_dict(self): return asdict(self)

@dataclass
class AnalyserUI:
    fmin: float
    fmax: float
    peak_db: float
    fft_size: int
    target_fps: int
    smoothing_tc: float
    def to_dict(self): return asdict(self)

@dataclass
class TuningUI:
    a4_lo: float
    a4_hi: float
    a4_step_coarse: float
    a4_step_fine: float
    fine_span_hz: float
    def to_dict(self): return asdict(self)

