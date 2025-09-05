from __future__ import annotations
import json
import streamlit as st
import numpy as np
from tuning_generator import pack_defaults
from modules.audio import decode_for_readout, to_wav_b64
from modules.peaks import extract_global_peaks
from modules.tuning import coarse_to_fine
from modules.spiral import build_spiral_params, render_html_realtime
from modules.config import SpiralUI, RippleUI, AnalyserUI, TuningUI

st.set_page_config(page_title="Tuning Analyser — TRUE Spiral (Realtime)", layout="wide")
st.title(" Tuning Analyser — TRUE Spiral (Realtime, modular)")

with st.sidebar:
    st.header("Realtime Spiral")
    spiral_ui = SpiralUI(
        turns=st.slider("Octave span (turns)", 2, 10, 4),
        bins_per_turn=st.slider("Resolution (bins/turn)", 180, 1440, 720, step=60),
        spokes=st.slider("Reference spokes (per 100)", 6, 36, 12, step=1),
        line_width=st.slider("Spiral Line Width", 1, 6, 3, step=1),
        max_segments=st.slider("Colored Line Segments", 60, 500, 300, step=20),
    )

    st.subheader("Ripple Controls (cardiogram-smooth)")
    ripple_ui = RippleUI(
        wiggle_gain=st.slider("Ripple Gain ()", 0.00, 2.00, 0.60, step=0.01),
        ripple_target=st.slider("Ripple Target (radius units)", 0.00, 1.50, 0.45, step=0.01),
        ripple_max=st.slider("Ripple Max (hard cap)", 0.05, 2.00, 0.90, step=0.05),
        ripple_gamma=st.slider("Ripple Gamma (contrast)", 0.40, 1.60, 0.85, step=0.01),
        ripple_temporal_cutoff=st.slider("Ripple Temporal Cutoff (Hz)", 0.2, 12.0, 3.0, step=0.1),
        smooth_bins=st.slider("Ripple Smoothing (bins)", 1, 81, 21, step=2),
    )

    st.header("WebAudio Analyser (live)")
    analyser_ui = AnalyserUI(
        fmin=st.number_input("Min freq (Hz)", 20.0, 400.0, 60.0, step=5.0),
        fmax=st.number_input("Max freq (Hz)", 500.0, 12000.0, 2200.0, step=50.0),
        peak_db=st.slider("Magnitude threshold (dBFS)", -120, -10, -80, step=2),
        fft_size=st.selectbox("FFT size", [2048, 4096, 8192, 16384], index=1),
        target_fps=st.slider("Target FPS", 10, 60, 30, step=5),
        smoothing_tc=st.slider("WebAudio smoothing (0..1)", 0.0, 0.95, 0.50, step=0.01),
    )

    st.header("Tuning Readout (one-shot)")
    systems_all = pack_defaults()
    default_keys = ["12-EDO","Pythagorean_12","Meantone_12_0.25comma","JI_5limit_chromatic_12"]
    choose = st.multiselect("Systems to test", list(systems_all.keys()), default=default_keys)
    tuning_ui = TuningUI(
        a4_lo=st.number_input("A4 low (Hz)", 400.0, 480.0, 430.0, step=0.5),
        a4_hi=st.number_input("A4 high (Hz)", 420.0, 500.0, 450.0, step=0.5),
        a4_step_coarse=st.number_input("A4 coarse step (Hz)", 0.2, 5.0, 1.0, step=0.2),
        a4_step_fine=st.number_input("A4 fine step (Hz)", 0.05, 1.0, 0.1, step=0.05),
        fine_span_hz=st.number_input("A4 fine  span (Hz)", 0.5, 5.0, 2.0, step=0.5),
    )

systems = {k: systems_all[k] for k in choose} if choose else systems_all
uploaded = st.file_uploader("Upload audio (WAV/FLAC/OGG/MP3)", type=["wav","flac","ogg","mp3"])

if uploaded is None:
    st.info("Upload audio to see a realtime, colored spiral with cardiogram-smooth ripples.")
    st.stop()

raw_in = uploaded.read()

# decode for readout
y, sr = decode_for_readout(raw_in)
if y is None:
    st.error("Audio decode failed."); st.stop()

# estimate tuning once
all_f, all_m = extract_global_peaks(y, sr, analyser_ui.fmin, analyser_ui.fmax, analyser_ui.peak_db)
if all_f.size == 0:
    st.warning("No usable peaks for tuning readout; using A4=440 Hz.")
    best = [{"name":"12-EDO","a4":440.0,"offset":0.0,"mad_cents":100.0}]
else:
    best = coarse_to_fine(all_f, all_m, systems, tuning_ui)

st.subheader("Best tuning matches")
for i, r in enumerate(best, 1):
    st.markdown(f"**{i}. {r['name']}** — A4  **{r['a4']:.2f} Hz**, tonic offset **{r['offset']:.1f}**, error  **{r['mad_cents']:.1f}**")

a4_ref = float(best[0]["a4"])

# geometry & realtime HTML
spiral_params = build_spiral_params(a4_ref, spiral_ui)
audio_b64 = to_wav_b64(raw_in)
html = render_html_realtime(spiral_params, ripple_ui, analyser_ui)
st.components.v1.html(html.replace("{{AUDIO_B64}}", audio_b64)
                        .replace("{{PARAMS_JSON}}", json.dumps(spiral_params | ripple_ui.to_dict() | analyser_ui.to_dict())),
                      height=720)

