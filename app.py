# app.py
from __future__ import annotations
import base64, io, json
import numpy as np
import streamlit as st
import soundfile as sf
from tuning_generator import pack_defaults

# ─────────────────────────── Streamlit UI ───────────────────────────
st.set_page_config(page_title="Tuning Analyser — TRUE Spiral (Realtime)", layout="wide")
st.title("🎼 Tuning Analyser — TRUE Spiral (Realtime)")

with st.sidebar:
    st.header("Realtime Spiral")
    turns          = st.slider("Octave span (turns)", 2, 10, 4)
    bins_per_turn  = st.slider("Resolution (bins/turn)", 180, 1440, 720, step=60)
    spokes         = st.slider("Reference spokes (per 100¢)", 6, 36, 12, step=1)
    line_width     = st.slider("Spiral Line Width", 1, 6, 3, step=1)
    max_segments   = st.slider("Colored Line Segments", 60, 600, 320, step=20)
    show_trail     = st.checkbox("Trail / afterglow", True)
    trail_frames   = st.slider("Trail length (frames)", 2, 24, 10, step=1)

    st.subheader("Ripple Controls")
    ripple_mode    = st.radio("Ripple Mode", ["Energy", "Grid-locked"], index=0,
                              help="Energy: raw spectral energy • Grid-locked: energy weighted by detected tuning grid")
    wiggle_gain    = st.slider("Ripple Gain (×)", 0.00, 2.00, 0.70, step=0.01)
    ripple_target  = st.slider("Ripple Target (radius units)", 0.00, 1.50, 0.45, step=0.01)
    ripple_max     = st.slider("Ripple Max (hard cap)", 0.05, 2.00, 0.90, step=0.05)
    ripple_gamma   = st.slider("Ripple Gamma (contrast)", 0.40, 1.60, 0.90, step=0.01)
    smooth_bins    = st.slider("Ripple Smoothing (bins)", 1, 81, 21, step=2)
    ripple_fc      = st.slider("Ripple Temporal Cutoff (Hz)", 0.2, 12.0, 3.0, step=0.1)

    st.header("WebAudio (live FFT)")
    fmin           = st.number_input("Min freq (Hz)", 20.0, 400.0, 60.0, step=5.0)
    fmax           = st.number_input("Max freq (Hz)", 500.0, 12000.0, 2200.0, step=50.0)
    peak_db        = st.slider("Magnitude threshold (dBFS)", -120, -10, -80, step=2)
    fft_size       = st.selectbox("FFT size", [2048, 4096, 8192, 16384], index=1)  # 4096
    target_fps     = st.slider("Target FPS", 10, 60, 30, step=5)
    smoothing_tc   = st.slider("WebAudio smoothing (0..1)", 0.0, 0.95, 0.50, step=0.01)

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

# ───────────────────── Server-side helpers (tuning) ─────────────────────
def amplitude_to_db(x, ref=1.0, amin=1e-12):
    x = np.maximum(x, amin); return 20.0 * np.log10(x / ref)

def extract_global_peaks(y, sr, fmin, fmax, peak_db):
    # Medium STFT; collect top peaks per frame → pool
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
        pk, props = find_peaks(spec, height=peak_db)
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

# ─────────────────────────── Main flow ───────────────────────────
if uploaded is None:
    st.info("Upload audio to see a realtime, colored spiral with cardiogram-smooth ripples.")
    st.stop()

raw_in = uploaded.read()

# Tuning estimation (server)
try:
    y, sr = sf.read(io.BytesIO(raw_in), dtype='float32', always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
except Exception as e:
    st.error(f"Decode failed: {e}")
    st.stop()

all_f, all_m = extract_global_peaks(y, sr, fmin, fmax, peak_db)
if all_f.size == 0:
    st.warning("No usable peaks for tuning readout; using A4=440 Hz.")
    best = [{"name":"12-EDO","a4":440.0,"offset":0.0,"mad_cents":100.0}]
else:
    best = coarse_to_fine(all_f, all_m, systems, a4_lo, a4_hi, a4_step_coarse, a4_step_fine, fine_span_hz)

st.subheader("Best tuning matches")
for i, r in enumerate(best, 1):
    st.markdown(
        f"**{i}. {r['name']}** — A4 ≈ **{r['a4']:.2f} Hz**, tonic offset **{r['offset']:.1f}¢**, error ≈ **{r['mad_cents']:.1f}¢**"
    )

# Spiral geometry (server → JS)
a4_ref = float(best[0]["a4"])
span_turns = int(turns)
bins_total = int(bins_per_turn * span_turns)
half       = span_turns / 2.0
phi_grid   = np.linspace(-half, +half, bins_total, endpoint=False).astype(float)
r_base     = phi_grid + half + 0.25
theta0     = 2.0*np.pi*phi_grid
x0, y0     = r_base*np.cos(theta0), r_base*np.sin(theta0)
Rmax       = float(np.max(np.sqrt(x0**2+y0**2)) + 0.75)

# segments (colored line chunks)
chunk = max(1, min(len(phi_grid)//2, int(np.ceil(len(phi_grid) / max_segments))))
seg_indices = []
ii = 0
while ii < len(phi_grid)-1:
    jj = min(len(phi_grid)-1, ii + chunk)
    seg_indices.append((ii, jj))
    ii = jj

# grid cents of best system for grid-locked mode
best_system_name = best[0]["name"]
grid_cents       = [n["cents"] for n in systems[best_system_name]["notes"]]

params = dict(
    phi_grid=phi_grid.tolist(),
    r_base=r_base.tolist(),
    seg_idx=seg_indices,
    Rmax=Rmax,
    half=half,
    spokes=int(spokes),
    line_w=int(line_width),
    show_trail=bool(show_trail),
    trail_frames=int(trail_frames),

    # Ripple control & analyser
    ripple_mode=str(ripple_mode),
    wiggle_gain=float(wiggle_gain),
    ripple_target=float(ripple_target),
    ripple_max=float(ripple_max),
    ripple_gamma=float(ripple_gamma),
    smooth_bins=int(smooth_bins),
    ripple_fc=float(ripple_fc),

    fmin=float(fmin), fmax=float(fmax), peak_db=float(peak_db),
    fft_size=int(fft_size), target_fps=int(target_fps), smoothing_tc=float(smoothing_tc),

    # Reference
    a4_ref=a4_ref,
    best_system=best_system_name,
    grid_cents=grid_cents,
)

audio_b64 = to_wav_b64(raw_in)

# Export session JSON
st.download_button(
    "⬇️ Export session JSON",
    file_name="tuning_session.json",
    mime="application/json",
    data=json.dumps({"tuning_best": best, "params": params}, indent=2)
)

# ───────────────────── HTML + JS: realtime spiral ─────────────────────
st.components.v1.html(f"""
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.31.1.min.js"></script>
  <style>
    body {{ background:#0a0a0a; margin:0; }}
    .wrap {{ display:flex; gap:16px; flex-direction:column; }}
    #plot {{ width:100%; height:560px; }}
    .row {{ display:flex; gap:12px; align-items:center; }}
    audio {{ width:100%; outline:none; }}
    button {{ background:#111; color:#ddd; border:1px solid #333; padding:6px 12px; border-radius:6px; }}
    button:hover {{ background:#222; }}
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
    // ───── Params from Python ─────
    const P = {json.dumps(params)};

    const phi_grid    = Float64Array.from(P.phi_grid);
    const r_base      = Float64Array.from(P.r_base);
    const seg_idx     = P.seg_idx;
    const Rmax        = P.Rmax;
    const HALF        = P.half;
    const SPOKES      = P.spokes;
    const LINE_W      = P.line_w;
    const SHOW_TRAIL  = !!P.show_trail;
    const TRAIL_FR    = P.trail_frames|0;

    const RIPPLE_MODE = (P.ripple_mode || "Energy");
    const WGAIN       = P.wiggle_gain;
    const RIP_TGT     = P.ripple_target;
    const RIP_MAX     = P.ripple_max;
    const RIP_GAM     = P.ripple_gamma;
    const SMOOTH      = P.smooth_bins|0;
    const RIP_FC      = Math.max(0.01, P.ripple_fc);

    const FMIN     = P.fmin, FMAX = P.fmax;
    const PKDB     = P.peak_db;
    const FFTSIZE  = P.fft_size|0;
    const TARGET_FPS = P.target_fps|0;
    const SMOOTH_TC  = Math.min(0.5, Math.max(0.0, P.smoothing_tc));

    const A4        = P.a4_ref;
    const GRID_CENTS= P.grid_cents || [];

    // ───── Audio: build WAV blob → MediaElement → WebAudio ─────
    const b64 = "{audio_b64}";
    function b64ToBlob(b64Data, contentType) {{
      const byteChars = atob(b64Data);
      const byteNums = new Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
      const byteArray = new Uint8Array(byteNums);
      return new Blob([byteArray], {{ type: "audio/wav" }});
    }}
    const aud = document.getElementById('aud');
    try {{
      const blob = b64ToBlob(b64, "audio/wav");
      const url = URL.createObjectURL(blob);
      aud.src = url;
    }} catch (e) {{ console.error("Audio blob creation failed:", e); }}

    const AC = new (window.AudioContext || window.webkitAudioContext)();
    const src = AC.createMediaElementSource(aud);
    const analyser = AC.createAnalyser();
    analyser.fftSize = FFTSIZE;
    analyser.smoothingTimeConstant = SMOOTH_TC;
    src.connect(analyser); analyser.connect(AC.destination);

    const bins = analyser.frequencyBinCount;
    const freqData = new Float32Array(bins);
    function binFreq(i) {{ return (i * AC.sampleRate * 0.5) / bins; }}

    // ───── Plotly init: guides + colored segments seeded with base coords ─────
    const traces = [];
    // spokes
    for (let s=0; s<SPOKES; s++) {{
      const ang = 2*Math.PI * s / SPOKES;
      traces.push({{
        type:'scatter', mode:'lines',
        x:[-Rmax*Math.cos(ang), Rmax*Math.cos(ang)],
        y:[-Rmax*Math.sin(ang), Rmax*Math.sin(ang)],
        line:{{width:1, color:'rgba(255,255,255,0.08)'}},
        hoverinfo:'skip', showlegend:false
      }});
    }}
    // octave circles
    for (let k=Math.floor(-HALF); k<=Math.ceil(HALF); k++) {{
      const xx = [], yy = [];
      for (let i=0;i<=360;i++) {{
        const a = i*Math.PI/180; const r = k + HALF + 0.25;
        xx.push(r*Math.cos(a)); yy.push(r*Math.sin(a));
      }}
      traces.push({{
        type:'scatter', mode:'lines',
        x:xx, y:yy, line:{{width:1, color:'rgba(255,255,255,0.08)'}},
        hoverinfo:'skip', showlegend:false
      }});
    }}

    // base spiral (visible immediately)
    const theta = new Float64Array(phi_grid.length);
    for (let i=0;i<phi_grid.length;i++) theta[i] = 2*Math.PI*phi_grid[i];
    const xbase = new Float64Array(phi_grid.length);
    const ybase = new Float64Array(phi_grid.length);
    for (let i=0;i<phi_grid.length;i++) {{
      const r = r_base[i];
      xbase[i] = r * Math.cos(theta[i]);
      ybase[i] = r * Math.sin(theta[i]);
    }}

    function hsvToRgbHex(h, s, v) {{
      const i = Math.floor(h*6)%6, f = h*6 - i;
      const p = v*(1-s), q = v*(1-f*s), t = v*(1-(1-f)*s);
      let r,g,b;
      if (i===0) {{ r=v; g=t; b=p; }}
      else if (i===1) {{ r=q; g=v; b=p; }}
      else if (i===2) {{ r=p; g=v; b=t; }}
      else if (i===3) {{ r=p; g=q; b=v; }}
      else if (i===4) {{ r=t; g=p; b=v; }}
      else           {{ r=v; g=p; b=q; }}
      const R=(r*255)|0, G=(g*255)|0, B=(b*255)|0;
      return "#" + R.toString(16).padStart(2,'0') + G.toString(16).padStart(2,'0') + B.toString(16).padStart(2,'0');
    }}

    const segStart = traces.length;
    for (const [i,j] of seg_idx) {{
      const mid = 0.5*(phi_grid[i] + phi_grid[j]);
      const hue = ((mid % 1)+1)%1;
      const col = hsvToRgbHex(hue, 0.95, 0.95);
      traces.push({{
        type:'scatter', mode:'lines',
        x: Array.from(xbase.slice(i, j+1)),
        y: Array.from(ybase.slice(i, j+1)),
        line: {{ width: LINE_W, color: col }},
        hoverinfo:'skip', showlegend:false
      }});
    }}

    const layout = {{
      template:'plotly_dark',
      paper_bgcolor:'#0a0a0a',
      plot_bgcolor:'#0a0a0a',
      xaxis:{{ visible:false, range:[-Rmax, Rmax] }},
      yaxis:{{ visible:false, range:[-Rmax, Rmax], scaleanchor:'x', scaleratio:1 }},
      margin:{{ l:10, r:10, t:40, b:10 }},
      title:"TRUE Spiral — angle: pitch class • radius: octaves (colored; LIVE cardiogram ripples)",
    }};
    const plotDiv = document.getElementById('plot');
    Plotly.newPlot(plotDiv, traces, layout);

    // ───── Utilities ─────
    function smoothCircular(arr, win) {{
      if (win<=1) return arr.slice();
      const n = arr.length, pad = Math.floor(win/2), out = new Float64Array(n);
      let acc = 0;
      for (let k=-pad; k<=pad; k++) acc += arr[(k+n)%n];
      for (let i=0; i<n; i++) {{
        out[i] = acc / (2*pad+1);
        const outIdx = (i-pad+n)%n, inIdx = (i+pad+1)%n;
        acc += arr[inIdx] - arr[outIdx];
      }}
      return Array.from(out);
    }}

    function gridLockedSalience(energy, phi_grid, centsGrid) {{
      const pc = new Float64Array(phi_grid.length);
      for (let i=0; i<phi_grid.length; i++) {{
        let v = (1200.0 * (phi_grid[i] % 1.0));
        if (v < 0) v += 1200.0;
        pc[i] = v;
      }}
      const kappa = 1.0/(30.0*30.0); // ~30c Gaussian
      const out = new Float64Array(energy.length);
      for (let i=0; i<energy.length; i++) {{
        const pci = pc[i];
        let acc = 0.0;
        for (let j=0; j<centsGrid.length; j++) {{
          let d = Math.abs(pci - centsGrid[j]) % 1200.0;
          d = Math.min(d, 1200.0 - d);
          const w = Math.exp(-kappa * d * d);
          acc += w;
        }}
        out[i] = energy[i] * acc;
      }}
      return out;
    }}

    // ───── Tick loop ─────
    const x = new Float64Array(phi_grid.length);
    const y = new Float64Array(phi_grid.length);
    const trail = [];
    let lastTS = 0;

    function tick(ts) {{
      if (!lastTS) lastTS = ts;
      const dt = (ts - lastTS) / 1000.0;
      const minDt = 1.0 / Math.max(10, TARGET_FPS);
      if (dt < minDt) {{ requestAnimationFrame(tick); return; }}
      lastTS = ts;

      if (!aud.paused) {{
        analyser.getFloatFrequencyData(freqData); // dBFS negatives

        // Accumulate spectral power → phi bins
        const energy = new Float64Array(phi_grid.length);
        for (let bi=0; bi<bins; bi++) {{
          const db = freqData[bi];
          if (!isFinite(db) || db < PKDB) continue;
          const f = binFreq(bi);
          if (f < FMIN || f > FMAX) continue;
          const phi = Math.log2(f / A4);
          if (phi < -HALF || phi >= HALF) continue;
          const idx = Math.floor((phi + HALF) / (2*HALF) * phi_grid.length);
          const ii = Math.max(0, Math.min(phi_grid.length-1, idx));
          const pwr = Math.pow(10, db/10);
          energy[ii] += pwr;
        }}

        // Robust normalize
        const tmp = Array.from(energy).sort((a,b)=>a-b);
        const q = (t)=> tmp[Math.max(0, Math.min(tmp.length-1, Math.floor(t*(tmp.length-1))))];
        const med = q(0.5), p95 = q(0.95);
        let scale = p95 - med; if (scale <= 1e-12) scale = 1e-12;
        for (let i=0;i<energy.length;i++) energy[i] = (energy[i] - med) / scale;

        // Temporal LPF (cardiogram feel)
        const alpha = 1 - Math.exp(-2*Math.PI*RIP_FC*dt);
        if (!window._energyLP || window._energyLP.length !== energy.length) {{
          window._energyLP = new Float64Array(energy.length);
        }}
        const energyLP = window._energyLP;
        for (let i=0;i<energy.length;i++) {{
          energyLP[i] += alpha * (energy[i] - energyLP[i]);
        }}

        // Gamma
        if (Math.abs(RIP_GAM - 1.0) > 1e-6) {{
          for (let i=0;i<energyLP.length;i++) {{
            const s = Math.sign(energyLP[i]), a = Math.abs(energyLP[i]);
            energyLP[i] = s * Math.pow(a, RIP_GAM);
          }}
        }}

        // Double spatial smooth
        const sm1 = (SMOOTH>1) ? smoothCircular(Array.from(energyLP), SMOOTH) : Array.from(energyLP);
        const energySm = (SMOOTH>1) ? smoothCircular(sm1, SMOOTH) : sm1;

        // Choose ripple drive
        let drive = energySm;
        if (RIPPLE_MODE === "Grid-locked" && GRID_CENTS.length > 0) {{
          drive = gridLockedSalience(energySm, phi_grid, GRID_CENTS);
        }}

        // Zero-center & auto-gain
        let mean = 0; for (let i=0;i<drive.length;i++) mean += drive[i]; mean /= drive.length;
        for (let i=0;i<drive.length;i++) drive[i] -= mean;

        let rms = 0; for (let i=0;i<drive.length;i++) rms += drive[i]*drive[i];
        rms = Math.sqrt(rms / drive.length);
        let gain = (rms > 1e-9) ? (RIP_TGT / rms) : 0.0;
        gain *= Math.max(0.0, WGAIN);
        const RCLAMP = Math.max(0.01, RIP_MAX);

        // Coords
        for (let i=0;i<phi_grid.length;i++) {{
          const dr = Math.max(-RCLAMP, Math.min(RCLAMP, gain * drive[i]));
          const r = r_base[i] + dr;
          x[i] = r * Math.cos(theta[i]);
          y[i] = r * Math.sin(theta[i]);
        }}

        // Trail buffer
        if (SHOW_TRAIL) {{
          trail.unshift({{ x: Array.from(x), y: Array.from(y) }});
          if (trail.length > TRAIL_FR) trail.pop();
        }} else {{
          trail.length = 0;
        }}

        // Update colored segments (head)
        const updateX = [], updateY = [], idxs = [];
        for (let s=0; s<seg_idx.length; s++) {{
          const [i0, j0] = seg_idx[s];
          updateX.push(Array.from(x.slice(i0, j0+1)));
          updateY.push(Array.from(y.slice(i0, j0+1)));
          idxs.push(segStart + s);
        }}
        Plotly.update(plotDiv, {{x: updateX, y: updateY}}, {{}}, idxs);

        // Quick trail redraws (fading)
        if (SHOW_TRAIL && trail.length > 1) {{
          if (!window._segColors) {{
            window._segColors = [];
            for (let s=0; s<seg_idx.length; s++) {{
              window._segColors.push(plotDiv.data[segStart + s].line.color);
            }}
          }}
          function colorWithAlpha(hex, a) {{
            const r = parseInt(hex.slice(1,3),16),
                  g = parseInt(hex.slice(3,5),16),
                  b = parseInt(hex.slice(5,7),16);
            return `rgba(${{r}},${{g}},${{b}},${{a.toFixed(3)}})`;
          }}
          for (let k=1; k<trail.length; k++) {{
            const fade = Math.max(0.15, 1.0 - (k/(TRAIL_FR+0.5)));
            const uX = [], uY = [], uCol = [], i2 = [];
            for (let s=0; s<seg_idx.length; s++) {{
              const [i0, j0] = seg_idx[s];
              const fr = trail[k];
              uX.push(fr.x.slice(i0, j0+1));
              uY.push(fr.y.slice(i0, j0+1));
              uCol.push(colorWithAlpha(window._segColors[s], fade));
              i2.push(segStart + s);
            }}
            Plotly.update(plotDiv, {{ x: uX, y: uY, "line.color": uCol }}, {{}}, i2);
          }}
        }}
      }}
      requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);

    // Controls
    document.getElementById('play').addEventListener('click', async () => {{
      try {{ if (AC.state === 'suspended') await AC.resume(); aud.play(); }} catch(e){{ console.error(e); }}
    }});
    document.getElementById('pause').addEventListener('click', () => aud.pause());
  </script>
</body>
</html>
""", height=720)

