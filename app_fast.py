# app.py
from __future__ import annotations
import io, json, base64
import numpy as np
import streamlit as st
import soundfile as sf
from tuning_generator import pack_defaults

# ========================= UI =========================
st.set_page_config(page_title="TRUE Spiral — Realtime (FAST, Smooth)", layout="wide")
st.title("🎼 TRUE Spiral — Realtime (FAST, Smooth)")

with st.sidebar:
    st.header("Spiral Geometry")
    turns          = st.slider("Octave span (turns)", 2, 10, 4)
    bins_per_turn  = st.slider("Resolution (bins/turn)", 180, 1440, 720, step=60)
    point_cap      = st.slider("Max live points (cap)", 500, 12000, 4000, step=100,
                               help="Upper bound on spiral vertices per frame (after oversample).")
    spokes         = st.slider("Reference spokes (per 100¢)", 6, 36, 12, step=1)
    line_width     = st.slider("Live line width (px)", 1, 6, 3, step=1)

    st.subheader("Ripple (cardiogram feel)")
    wiggle_gain    = st.slider("Ripple Gain (×)", 0.00, 2.00, 0.60, step=0.01)
    ripple_target  = st.slider("Ripple Target (radius units)", 0.00, 1.50, 0.45, step=0.01)
    ripple_max     = st.slider("Ripple Max (clamp)", 0.05, 2.00, 0.90, step=0.05)
    ripple_gamma   = st.slider("Ripple Gamma (contrast)", 0.40, 1.60, 0.90, step=0.01)
    smooth_bins    = st.slider("Spatial smoothing (bins)", 1, 81, 21, step=2)
    ripple_fc      = st.slider("Temporal cutoff (Hz)", 0.2, 12.0, 3.0, step=0.1)
    oversample     = st.slider("Oversample φ (1..4)", 1, 4, 3, step=1,
                               help="Cosine interpolation between bins; 2–3 looks very smooth.")

    st.subheader("Analyser (WebAudio)")
    fmin           = st.number_input("Min freq (Hz)", 20.0, 400.0, 60.0, step=5.0)
    fmax           = st.number_input("Max freq (Hz)", 500.0, 12000.0, 2200.0, step=50.0)
    fft_size       = st.selectbox("FFT size", [2048, 4096, 8192], index=1)
    smoothing_tc   = st.slider("Analyser smoothing (0..1)", 0.0, 0.95, 0.30, step=0.01)
    hq_ripples     = st.checkbox("High-quality ripples (float dB)", True,
                                 help="getFloatFrequencyData (smoother). Off = bytes (faster).")
    byte_floor     = st.slider("Byte floor (0..255)", 0, 255, 140, step=5,
                               help="Ignored when HQ is on. Higher skips more quiet bins.")

    st.subheader("Color / Mode")
    live_color     = st.color_picker("Live line color", "#39FF14")
    bg_color       = st.color_picker("Background", "#0a0a0a")
    show_segments_when_paused = st.checkbox("Show colored segments when paused", True,
        help="Multi-color spiral drawn once when paused (not during playback).")

    st.header("Tuning Readout (server)")
    systems_all    = pack_defaults()
    default_keys   = ["12-EDO","Pythagorean_12","Meantone_12_0.25comma","JI_5limit_chromatic_12"]
    choose         = st.multiselect("Systems to test", list(systems_all.keys()), default=default_keys)
    a4_lo          = st.number_input("A4 low (Hz)", 400.0, 480.0, 430.0, step=0.5)
    a4_hi          = st.number_input("A4 high (Hz)", 420.0, 500.0, 450.0, step=0.5)
    a4_step_coarse = st.number_input("A4 coarse step (Hz)", 0.2, 5.0, 1.0, step=0.2)
    a4_step_fine   = st.number_input("A4 fine step (Hz)", 0.05, 1.0, 0.1, step=0.05)
    fine_span_hz   = st.number_input("A4 fine ± span (Hz)", 0.5, 5.0, 2.0, step=0.5)

systems = {k: systems_all[k] for k in choose} if choose else systems_all
uploaded = st.file_uploader("Upload audio (WAV/FLAC/OGG/MP3)", type=["wav","flac","ogg","mp3"])

# ========================= Helpers =========================
def amplitude_to_db(x, ref=1.0, amin=1e-12):
    x = np.maximum(x, amin); return 20.0 * np.log10(x / ref)

def extract_global_peaks(y, sr, fmin, fmax, peak_db_dbfs=-80.0):
    # Light STFT → pooled peaks for tuning readout
    from scipy.signal import stft, find_peaks
    nperseg = int(sr * 0.064); hop = int(sr * 0.024)
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
        pk, props = find_peaks(spec, height=peak_db_dbfs)
        if pk.size == 0: continue
        h = props["peak_heights"]
        sel = np.argsort(h)[-12:][::-1]
        all_f.append(fbin[pk[sel]]); all_m.append(h[sel])
    if all_f:
        return np.concatenate(all_f), np.concatenate(all_m)
    return np.array([]), np.array([])

def score_system(obs_freqs, obs_mags, cents_grid, a4_ref):
    obs_pc = (1200.0 * np.log2(obs_freqs / a4_ref)) % 1200.0
    w = np.maximum(obs_mags - obs_mags.min(), 1.0)
    best = (1e9, 0.0)
    for off in cents_grid:
        grid = (cents_grid - off) % 1200.0
        diffs = np.abs(obs_pc[:, None] - grid[None, :]) % 1200.0
        diffs = np.minimum(diffs, 1200.0 - diffs)
        dmin = diffs.min(axis=1)
        order = np.argsort(dmin)
        w_sorted = w[order]; d_sorted = dmin[order]
        cumw = np.cumsum(w_sorted)
        med_idx = np.searchsorted(cumw, cumw[-1] / 2.0)
        mad = d_sorted[min(med_idx, len(d_sorted)-1)]
        if mad < best[0]: best = (float(mad), float(off))
    return {"offset": best[1], "mad_cents": best[0]}

def coarse_to_fine(obs_freqs, obs_mags, systems_dict, a4_lo, a4_hi, step_coarse, step_fine, fine_span):
    results, coarse = [], []
    for name, sys in systems_dict.items():
        cents = np.array([n["cents"] for n in sys["notes"]], dtype=float)
        best = {"a4": None, "mad": 1e9, "offset": 0.0}
        a = a4_lo
        while a <= a4_hi + 1e-9:
            sc = score_system(obs_freqs, obs_mags, cents, a)
            if sc["mad_cents"] < best["mad"]:
                best = {"a4": float(a), "mad": sc["mad_cents"], "offset": sc["offset"]}
            a += step_coarse
        coarse.append((name, best))
    coarse.sort(key=lambda x: x[1]["mad"])
    for name, bestc in coarse[:2]:
        cents = np.array([n["cents"] for n in systems_dict[name]["notes"]], dtype=float)
        a_vals = np.arange(bestc["a4"] - fine_span, bestc["a4"] + fine_span + 1e-9, step_fine, dtype=float)
        best = {"a4": None, "mad": 1e9, "offset": 0.0}
        for a in a_vals:
            sc = score_system(obs_freqs, obs_mags, cents, a)
            if sc["mad_cents"] < best["mad"]:
                best = {"a4": float(a), "mad": sc["mad_cents"], "offset": sc["offset"]}
        results.append({"name": name, "a4": best["a4"], "offset": best["offset"], "mad_cents": best["mad"]})
    results.sort(key=lambda x: x["mad_cents"])
    return results

def to_wav_b64(raw: bytes) -> str:
    y, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode('ascii')

# ========================= Main =========================
if uploaded is None:
    st.info("Upload audio to see a realtime WebGL spiral with **sine-smooth** ripples.")
    st.stop()

raw_in = uploaded.read()

# Tuning (one-shot)
try:
    y, sr = sf.read(io.BytesIO(raw_in), dtype='float32', always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)
except Exception as e:
    st.error(f"Decode failed: {e}")
    st.stop()

all_f, all_m = extract_global_peaks(y, sr, fmin, fmax)
if all_f.size == 0:
    st.warning("No usable peaks for tuning readout; using A4=440 Hz.")
    best = [{"name":"12-EDO","a4":440.0,"offset":0.0,"mad_cents":100.0}]
else:
    best = coarse_to_fine(all_f, all_m, systems, a4_lo, a4_hi,
                          a4_step_coarse, a4_step_fine, fine_span_hz)

st.subheader("Best tuning matches")
for i, r in enumerate(best, 1):
    st.markdown(
        f"**{i}. {r['name']}** — A4 ≈ **{r['a4']:.2f} Hz**, tonic offset **{r['offset']:.1f}¢**, "
        f"error ≈ **{r['mad_cents']:.1f}¢**"
    )

# Spiral geometry → JS
a4_ref    = float(best[0]["a4"])
span_turns= int(turns)
bins_total= int(bins_per_turn * span_turns)

# Cap pre-JS so oversample fits within point_cap
bins_total = max(60, min(bins_total, int(point_cap // max(1, oversample))))
half      = span_turns / 2.0
phi_grid  = np.linspace(-half, +half, bins_total, endpoint=False).astype(float)
r_base    = phi_grid + half + 0.25
theta0    = 2.0*np.pi*phi_grid
x0, y0    = r_base*np.cos(theta0), r_base*np.sin(theta0)
Rmax      = float(np.max(np.sqrt(x0**2 + y0**2)) + 0.75)

# Colored segments for paused state
seg_idx = []
if show_segments_when_paused:
    max_segments = max(60, bins_total // 32)
    chunk = max(1, int(np.ceil(bins_total / max_segments)))
    i = 0
    while i < bins_total - 1:
        j = min(bins_total - 1, i + chunk)
        seg_idx.append((i, j))
        i = j

params = dict(
    # geometry
    phi_grid=phi_grid.tolist(),
    r_base=r_base.tolist(),
    spokes=int(spokes),
    line_w=int(line_width),
    Rmax=Rmax,
    half=half,
    seg_idx=seg_idx,
    point_cap=int(point_cap),
    oversample=int(oversample),

    # ripple/analyser
    wiggle_gain=float(wiggle_gain),
    ripple_target=float(ripple_target),
    ripple_max=float(ripple_max),
    ripple_gamma=float(ripple_gamma),
    smooth_bins=int(smooth_bins),
    ripple_fc=float(ripple_fc),
    fmin=float(fmin), fmax=float(fmax),
    fft_size=int(fft_size),
    smoothing_tc=float(smoothing_tc),
    hq_ripples=bool(hq_ripples),
    byte_floor=int(byte_floor),

    # look
    live_color=str(live_color),
    bg_color=str(bg_color),
    show_segments_when_paused=bool(show_segments_when_paused),

    # tuning
    a4_ref=a4_ref,
)

audio_b64 = to_wav_b64(raw_in)

# Export session bundle
st.download_button(
    "⬇️ Export session JSON",
    file_name="tuning_session.json",
    mime="application/json",
    data=json.dumps({"tuning_best": best, "params": params}, indent=2)
)

# ========================= HTML/JS (placeholder template) =========================
html_tpl = """
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.31.1.min.js"></script>
  <style>
    body { background:__BG__; margin:0; }
    .wrap { display:flex; gap:16px; flex-direction:column; }
    #plot { width:100%; height:560px; }
    .row { display:flex; gap:12px; align-items:center; }
    audio { width:100%; outline:none; }
    button { background:#111; color:#ddd; border:1px solid #333; padding:6px 12px; border-radius:6px; }
    button:hover { background:#222; }
  </style>
</head>
<body>
  <div class="wrap">
    <div id="plot"></div>
    <div class="row">
      <button id="play">Play</button>
      <button id="pause">Pause</button>
      <audio id="aud" controls preload="auto" style="flex:1"></audio>
    </div>
  </div>

  <script>
    // ----- Parameters from Python -----
    const P = __PARAMS__;
    const phi_grid = Float64Array.from(P.phi_grid);
    const r_base   = Float64Array.from(P.r_base);
    const seg_idx  = P.seg_idx;
    const Rmax     = P.Rmax;
    const HALF     = P.half;
    const SPOKES   = P.spokes|0;
    const WGAIN    = P.wiggle_gain;
    const SMOOTH   = P.smooth_bins|0;
    const FMIN     = P.fmin;
    const FMAX     = P.fmax;
    const PKDB     = -160; // use robust float path below
    const FFTSIZE  = P.fft_size|0;
    const TARGET_FPS = 30; // stable 30 FPS
    const SMOOTH_TC = Math.min(0.9, Math.max(0.0, P.smoothing_tc));
    const A4       = P.a4_ref;
    const RIP_GAM  = P.ripple_gamma;
    const RIP_TGT  = P.ripple_target;
    const RIP_MAX  = P.ripple_max;
    const RIP_FC   = Math.max(0.01, P.ripple_fc);
    const LINE_W   = P.line_w|0;

    const LIVE_BASE_COLOR = P.live_color || "#39FF14"; // used for paused snap only

    // ----- Audio: rebuild WAV blob -----
    const b64 = "__AUDIO_B64__";
    function b64ToBlob(b64Data) {
      const bytes = atob(b64Data);
      const arr = new Uint8Array(bytes.length);
      for (let i=0;i<bytes.length;i++) arr[i] = bytes.charCodeAt(i);
      return new Blob([arr], { type: "audio/wav" });
    }
    const aud = document.getElementById('aud');
    try { aud.src = URL.createObjectURL(b64ToBlob(b64)); } catch(e) { console.error("Audio blob creation failed:", e); }

    // ----- WebAudio graph -----
    const AC = new (window.AudioContext || window.webkitAudioContext)();
    const src = AC.createMediaElementSource(aud);
    const analyser = AC.createAnalyser();
    analyser.fftSize = FFTSIZE;
    analyser.smoothingTimeConstant = SMOOTH_TC;
    src.connect(analyser);
    analyser.connect(AC.destination);

    const bins = analyser.frequencyBinCount;
    const freqFloat = new Float32Array(bins); // high-quality dB path
    function binFreq(i) { return (i * AC.sampleRate * 0.5) / bins; }

    // ----- Spiral base -----
    const theta = new Float64Array(phi_grid.length);
    for (let i=0;i<phi_grid.length;i++) theta[i] = 2*Math.PI*phi_grid[i];
    const xbase = new Float64Array(phi_grid.length);
    const ybase = new Float64Array(phi_grid.length);
    for (let i=0;i<phi_grid.length;i++) {
      const r = r_base[i];
      xbase[i] = r * Math.cos(theta[i]);
      ybase[i] = r * Math.sin(theta[i]);
    }

    // Helpers
    function hannKernel(win){
      const n = Math.max(1, win|0);
      if (n<=1) return new Float64Array([1]);
      const k = new Float64Array(n);
      for (let i=0;i<n;i++) k[i] = 0.5*(1-Math.cos(2*Math.PI*i/(n-1)));
      let s=0; for (let i=0;i<n;i++) s+=k[i];
      for (let i=0;i<n;i++) k[i]/=(s||1);
      return k;
    }
    function convolveCircular(arr, ker){
      const n=arr.length, m=ker.length, out=new Float64Array(n), half=(m>>1);
      for (let i=0;i<n;i++){
        let acc=0;
        for (let j=0;j<m;j++){
          const idx=(i + j - half + n) % n;
          acc += arr[idx]*ker[j];
        }
        out[i]=acc;
      }
      return out;
    }
    function hsvToRgba(h,s,v,a){
      const i=Math.floor(h*6)%6, f=h*6-i;
      const p=v*(1-s), q=v*(1-f*s), t=v*(1-(1-f)*s);
      let r,g,b;
      if (i===0){r=v;g=t;b=p;} else if (i===1){r=q;g=v;b=p;}
      else if (i===2){r=p;g=v;b=t;} else if (i===3){r=p;g=q;b=v;}
      else if (i===4){r=t;g=p;b=v;} else {r=v;g=p;b=q;}
      return `rgba(${(r*255)|0},${(g*255)|0},${(b*255)|0},${a})`;
    }

    // ----- Plotly init: guides -----
    const traces = [];
    // spokes
    for (let s=0; s<SPOKES; s++) {
      const ang = 2*Math.PI * s / SPOKES;
      traces.push({
        type:'scatter', mode:'lines',
        x:[-Rmax*Math.cos(ang), Rmax*Math.cos(ang)],
        y:[-Rmax*Math.sin(ang), Rmax*Math.sin(ang)],
        line:{width:1, color:'rgba(255,255,255,0.08)'},
        hoverinfo:'skip', showlegend:false
      });
    }
    // octave circles
    for (let k=Math.floor(-HALF); k<=Math.ceil(HALF); k++) {
      const xx = [], yy = [];
      for (let i=0;i<=360;i++) {
        const a = i*Math.PI/180; const r = k + HALF + 0.25;
        xx.push(r*Math.cos(a)); yy.push(r*Math.sin(a));
      }
      traces.push({
        type:'scatter', mode:'lines',
        x:xx, y:yy, line:{width:1, color:'rgba(255,255,255,0.08)'},
        hoverinfo:'skip', showlegend:false
      });
    }

    // ----- RGB live line: build N segments + glow -----
    const SEG_LIVE = 48;               // total rainbow segments
    const GLOW_W   = Math.max(2, LINE_W + 6);
    const HUE_ROT_SPEED = 0.03;        // hue rotation speed (rev/s)

    // Pre-create segment traces (glow underlay + bright line)
    const segFirstIdx = traces.length;
    for (let s=0; s<SEG_LIVE; s++) {
      const hue = s / SEG_LIVE;
      const colGlow = hsvToRgba(hue, 0.85, 1.0, 0.10);  // faint wide glow
      const colLine = hsvToRgba(hue, 0.95, 1.0, 1.00);  // bright segment line
      // glow
      traces.push({
        type:'scattergl', mode:'lines',
        x:[0,0], y:[0,0],
        line:{ width: GLOW_W, color: colGlow },
        hoverinfo:'skip', showlegend:false
      });
      // bright
      traces.push({
        type:'scattergl', mode:'lines',
        x:[0,0], y:[0,0],
        line:{ width: LINE_W, color: colLine },
        hoverinfo:'skip', showlegend:false
      });
    }
    const segCount = SEG_LIVE * 2; // glow + bright per segment

    const layout = {
      template:'plotly_dark',
      paper_bgcolor:'__BG__',
      plot_bgcolor:'__BG__',
      xaxis:{ visible:false, range:[-Rmax, Rmax] },
      yaxis:{ visible:false, range:[-Rmax, Rmax], scaleanchor:'x', scaleratio:1 },
      margin:{ l:10, r:10, t:40, b:10 },
      title:'TRUE Spiral — RGB live line (smooth ripples + glow)'
    };
    const plotDiv = document.getElementById('plot');
    Plotly.newPlot(plotDiv, traces, layout, {responsive:true});

    // ----- Ripple computation (same cardiogram pipeline) -----
    const ker = hannKernel(Math.max(1, SMOOTH|0));
    function computeXY(dt){
      analyser.getFloatFrequencyData(freqFloat); // high quality dB
      const N = phi_grid.length;
      const energy = new Float64Array(N);
      for (let bi=0; bi<bins; bi++){
        const db = freqFloat[bi];
        if (!isFinite(db) || db < PKDB) continue;
        const f = binFreq(bi);
        if (f < FMIN || f > FMAX) continue;
        const phi = Math.log2(f / A4);
        if (phi < -HALF || phi >= HALF) continue;
        const idx = Math.floor((phi + HALF) / (2*HALF) * N);
        const ii = Math.max(0, Math.min(N-1, idx));
        energy[ii] += Math.pow(10, db/10);
      }
      // robust normalize
      const tmp = Array.from(energy).sort((a,b)=>a-b);
      const q = (t)=> tmp[Math.max(0, Math.min(tmp.length-1, Math.floor(t*(tmp.length-1))))];
      const med = q(0.5), p95 = q(0.95);
      let scale = p95 - med; if (scale <= 1e-12) scale = 1e-12;
      for (let i=0;i<N;i++) energy[i] = (energy[i] - med) / scale;
      // temporal EMA
      const alpha = 1 - Math.exp(-2*Math.PI*RIP_FC*dt);
      if (!window._lp || window._lp.length !== N) window._lp = new Float64Array(N);
      const lp = window._lp;
      for (let i=0;i<N;i++) lp[i] += alpha * (energy[i] - lp[i]);
      // spatial double Hann
      const sm1 = (SMOOTH>1) ? convolveCircular(Array.from(lp), ker) : Array.from(lp);
      const sm2 = (SMOOTH>1) ? convolveCircular(sm1, ker) : sm1;
      // gamma
      if (Math.abs(RIP_GAM - 1.0) > 1e-6) {
        for (let i=0;i<N;i++){ const s=Math.sign(sm2[i]), a=Math.abs(sm2[i]); sm2[i]= s*Math.pow(a, RIP_GAM); }
      }
      // zero-center + auto-gain + clamp
      let mean=0; for (let i=0;i<N;i++) mean+=sm2[i]; mean/=N;
      for (let i=0;i<N;i++) sm2[i]-=mean;
      let rms=0; for (let i=0;i<N;i++) rms+=sm2[i]*sm2[i]; rms=Math.sqrt(rms/N);
      let gain = (rms>1e-9) ? (RIP_TGT / rms) : 0.0;
      gain *= Math.max(0.0, WGAIN);
      const RCLAMP = Math.max(0.01, RIP_MAX);

      // new coords
      const X = new Array(N), Y = new Array(N);
      for (let i=0;i<N;i++){
        const dr = Math.max(-RCLAMP, Math.min(RCLAMP, gain * sm2[i]));
        const r  = r_base[i] + dr;
        X[i] = r * Math.cos(theta[i]);
        Y[i] = r * Math.sin(theta[i]);
      }
      return {X, Y};
    }

    // ----- Frame loop: update RGB segments only -----
    let lastTS = 0;
    function tick(ts){
      if (!lastTS) lastTS = ts;
      const dt = Math.max(1/120, (ts - lastTS)/1000);
      const minDt = 1.0 / Math.max(10, TARGET_FPS);
      if (dt < minDt) { requestAnimationFrame(tick); return; }
      lastTS = ts;

      if (!aud.paused) {
        const {X, Y} = computeXY(dt);

        // split points into SEG_LIVE equal chunks
        const N = X.length;
        const segLen = Math.floor(N / SEG_LIVE);
        const updateX = [], updateY = [], idxs = [], lineColors = [], glowColors = [], widths = [];
        // base hue rotation over time for the whole wheel
        const hueShift = (HUE_ROT_SPEED * (ts/1000)) % 1;

        for (let s=0; s<SEG_LIVE; s++){
          const i0 = (s * segLen) | 0;
          const i1 = (s === SEG_LIVE-1) ? (N-1) : ((i0 + segLen) | 0);
          const xseg = X.slice(i0, i1+1);
          const yseg = Y.slice(i0, i1+1);

          // glow trace
          const glowIdx = segFirstIdx + (2*s);
          const hue = (s / SEG_LIVE + hueShift) % 1;
          const colGlow = hsvToRgba(hue, 0.85, 1.0, 0.10);
          updateX.push(xseg); updateY.push(yseg); idxs.push(glowIdx);
          lineColors.push(colGlow); widths.push(Math.max(2, LINE_W + 6));

          // bright trace
          const lineIdx = segFirstIdx + (2*s) + 1;
          const colLine = hsvToRgba(hue, 0.95, 1.0, 1.00);
          updateX.push(xseg); updateY.push(yseg); idxs.push(lineIdx);
          lineColors.push(colLine); widths.push(LINE_W);
        }

        Plotly.restyle(plotDiv, {
          x: updateX,
          y: updateY,
          "line.color": lineColors,
          "line.width": widths
        }, idxs);
      }

      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    // Controls
    document.getElementById('play').addEventListener('click', async () => {
      try { if (AC.state === 'suspended') await AC.resume(); aud.play(); } catch(e){ console.error(e); }
    });
    document.getElementById('pause').addEventListener('click', () => {
      aud.pause();
      // On pause, show a clean mono-color coil (your picker) so RGB segments stop moving visibly
      const N = phi_grid.length;
      const segLen = Math.floor(N / SEG_LIVE);
      const idxs = [], updateX=[], updateY=[], lineColors=[], widths=[];
      for (let s=0; s<SEG_LIVE; s++){
        const i0 = (s * segLen) | 0;
        const i1 = (s === SEG_LIVE-1) ? (N-1) : ((i0 + segLen) | 0);
        const xseg = Array.from(xbase.slice(i0, i1+1));
        const yseg = Array.from(ybase.slice(i0, i1+1));
        const glowIdx = segFirstIdx + (2*s);
        const lineIdx = glowIdx + 1;
        // glow stay faint of the same hue as base color (use greenish glow)
        idxs.push(glowIdx); updateX.push(xseg); updateY.push(yseg);
        lineColors.push('rgba(0,255,170,0.10)'); widths.push(Math.max(2, LINE_W + 6));
        // bright line in your chosen color
        idxs.push(lineIdx); updateX.push(xseg); updateY.push(yseg);
        lineColors.push(LIVE_BASE_COLOR); widths.push(LINE_W);
      }
      Plotly.restyle(plotDiv, {
        x: updateX, y: updateY,
        "line.color": lineColors,
        "line.width": widths
      }, idxs);
    });
  </script>

</body>
</html>
"""

html = (
    html_tpl
    .replace("__PARAMS__", json.dumps(params))
    .replace("__AUDIO_B64__", audio_b64)
    .replace("__BG__", bg_color)
)
st.components.v1.html(html, height=720)

