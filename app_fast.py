from __future__ import annotations
import base64, io, json
import numpy as np
import streamlit as st
import soundfile as sf
from tuning_generator import pack_defaults

st.set_page_config(page_title="Tuning Analyser — TRUE Spiral (FAST WebGL)", layout="wide")
st.title("🎼 Tuning Analyser — TRUE Spiral (FAST WebGL)")

with st.sidebar:
    st.header("Spiral Geometry")
    turns          = st.slider("Octave span (turns)", 2, 10, 4)
    bins_per_turn  = st.slider("Resolution (bins/turn)", 180, 1440, 720, step=60)
    decimate_max   = st.slider("Max points (performance cap)", 500, 8000, 2500, step=100,
                               help="Upper bound on spiral samples sent to the browser. Lower = faster.")
    spokes         = st.slider("Reference spokes (per 100)", 6, 36, 12, step=1)
    line_width     = st.slider("Line width (px)", 1, 6, 3, step=1)

    st.subheader("Ripple (cardiogram)")
    wiggle_gain    = st.slider("Ripple Gain ()", 0.00, 2.00, 0.60, step=0.01)
    ripple_target  = st.slider("Ripple Target (radius units)", 0.00, 1.50, 0.45, step=0.01)
    ripple_max     = st.slider("Ripple Max (hard cap)", 0.05, 2.00, 0.90, step=0.05)
    ripple_gamma   = st.slider("Ripple Gamma", 0.40, 1.60, 0.85, step=0.01)
    smooth_bins    = st.slider("Ripple Smoothing (bins)", 1, 81, 21, step=2)
    ripple_fc      = st.slider("Temporal cutoff (Hz)", 0.2, 12.0, 3.0, step=0.1)

    st.subheader("Analyser (WebAudio)")
    fmin           = st.number_input("Min freq (Hz)", 20.0, 400.0, 60.0, step=5.0)
    fmax           = st.number_input("Max freq (Hz)", 500.0, 12000.0, 2200.0, step=50.0)
    # Fast path uses byte data; threshold is in 0..255 (higher = louder)
    byte_floor     = st.slider("Byte floor (0..255)", 0, 255, 140, step=5,
                               help="Ignore bins below this magnitude (speeds up CPU).")
    fft_size       = st.selectbox("FFT size", [2048, 4096, 8192], index=1)
    target_fps     = st.slider("Target FPS", 10, 60, 30, step=5)
    smoothing_tc   = st.slider("WebAudio smoothing (0..1)", 0.0, 0.95, 0.3, step=0.01)

    st.subheader("Color / Mode")
    enable_segments_when_paused = st.checkbox("Show colored segments when paused", True,
        help="While playing, a single WebGL line is drawn. When paused, a multi-color segmented spiral is shown once.")
    line_color    = st.color_picker("Live line color", "#39FF14")
    bg_color      = st.color_picker("Background", "#0a0a0a")

    st.header("Tuning Readout (server)")
    systems_all    = pack_defaults()
    default_keys   = ["12-EDO","Pythagorean_12","Meantone_12_0.25comma","JI_5limit_chromatic_12"]
    choose         = st.multiselect("Systems to test", list(systems_all.keys()), default=default_keys)
    a4_lo          = st.number_input("A4 low (Hz)", 400.0, 480.0, 430.0, step=0.5)
    a4_hi          = st.number_input("A4 high (Hz)", 420.0, 500.0, 450.0, step=0.5)
    a4_step_coarse = st.number_input("A4 coarse step (Hz)", 0.2, 5.0, 1.0, step=0.2)
    a4_step_fine   = st.number_input("A4 fine step (Hz)", 0.05, 1.0, 0.1, step=0.05)
    fine_span_hz   = st.number_input("A4 fine  span (Hz)", 0.5, 5.0, 2.0, step=0.5)

systems = {k: systems_all[k] for k in choose} if choose else systems_all
uploaded = st.file_uploader("Upload audio (WAV/FLAC/OGG/MP3)", type=["wav","flac","ogg","mp3"])

# ---------- server helpers ----------
def amplitude_to_db(x, ref=1.0, amin=1e-12):
    x = np.maximum(x, amin); return 20.0 * np.log10(x / ref)

def extract_global_peaks(y, sr, fmin, fmax, peak_db_dbfs=-80.0):
    # simple readout; not used in the fast loop
    from scipy.signal import stft, find_peaks
    nperseg = int(sr*0.064); hop = int(sr*0.024)
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

# ---------- main ----------
if uploaded is None:
    st.info("Upload audio to see the FAST WebGL spiral.")
    st.stop()

raw_in = uploaded.read()
try:
    y, sr = sf.read(io.BytesIO(raw_in), dtype='float32', always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
except Exception as e:
    st.error(f"Decode failed: {e}")
    st.stop()

# Tuning readout (one-shot)
all_f, all_m = extract_global_peaks(y, sr, fmin, fmax)
if all_f.size == 0:
    st.warning("No usable peaks for tuning readout; using A4=440 Hz.")
    best = [{"name":"12-EDO","a4":440.0,"offset":0.0,"mad_cents":100.0}]
else:
    best = coarse_to_fine(all_f, all_m, systems, a4_lo, a4_hi, a4_step_coarse, a4_step_fine, fine_span_hz)
a4_ref = float(best[0]["a4"])

# Spiral geometry (decimate aggressively to keep <= decimate_max points)
span_turns = int(turns)
bins_total = int(bins_per_turn * span_turns)
bins_total = int(min(bins_total, decimate_max))  # cap points
half       = span_turns / 2.0
phi_grid   = np.linspace(-half, +half, bins_total, endpoint=False).astype(float)
r_base     = phi_grid + half + 0.25
theta0     = 2.0*np.pi*phi_grid
x0, y0     = r_base*np.cos(theta0), r_base*np.sin(theta0)
Rmax       = float(np.max(np.sqrt(x0**2+y0**2)) + 0.75)

# colored segments are only used when paused  compute indices once (coarse)
seg_indices = []
if enable_segments_when_paused:
    max_segments = int(max(60, bins_total//32))
    chunk = max(1, int(np.ceil(bins_total / max_segments)))
    i = 0
    while i < bins_total-1:
        j = min(bins_total-1, i + chunk)
        seg_indices.append((i, j))
        i = j

grid_cents = [n["cents"] for n in systems[best[0]["name"]]["notes"]]

params = dict(
    phi_grid=phi_grid.tolist(),
    r_base=r_base.tolist(),
    seg_idx=seg_indices,
    Rmax=Rmax,
    half=half,
    spokes=int(spokes),
    line_w=int(line_width),
    # ripple/analyser
    wiggle_gain=float(wiggle_gain),
    ripple_target=float(ripple_target),
    ripple_max=float(ripple_max),
    ripple_gamma=float(ripple_gamma),
    smooth_bins=int(smooth_bins),
    ripple_fc=float(ripple_fc),
    fmin=float(fmin), fmax=float(fmax),
    byte_floor=int(byte_floor),
    fft_size=int(fft_size),
    target_fps=int(target_fps),
    smoothing_tc=float(smoothing_tc),
    a4_ref=a4_ref,
    grid_cents=grid_cents,
    enable_segments_when_paused=bool(enable_segments_when_paused),
    line_color=str(line_color),
    bg_color=str(bg_color),
)

audio_b64 = to_wav_b64(raw_in)

# Export params
st.download_button(" Export session (JSON)",
    file_name="tuning_session_fast.json",
    mime="application/json",
    data=json.dumps({"tuning_best": best, "params": params}, indent=2))

# ---------- FAST WebGL HTML ----------
st.components.v1.html(f"""
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.31.1.min.js"></script>
  <style>
    body {{ background:{params['bg_color']}; margin:0; }}
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
    const P = {json.dumps(params)};
    const phi_grid = Float64Array.from(P.phi_grid);
    const r_base = Float64Array.from(P.r_base);
    const seg_idx = P.seg_idx;
    const Rmax = P.Rmax, HALF = P.half, SPOKES = P.spokes, LINE_W = P.line_w;
    const A4 = P.a4_ref, GRID_CENTS = P.grid_cents || [];
    const WGAIN = P.wiggle_gain, RIP_TGT = P.ripple_target, RIP_MAX = P.ripple_max;
    const RIP_GAM = P.ripple_gamma, SMOOTH = P.smooth_bins|0, RIP_FC = Math.max(0.01, P.ripple_fc);
    const FMIN = P.fmin, FMAX = P.fmax, BYTE_FLOOR = P.byte_floor|0;
    const FFTSIZE = P.fft_size|0, TARGET_FPS = P.target_fps|0, SMOOTH_TC = Math.min(0.8, Math.max(0.0, P.smoothing_tc));
    const ENABLE_SEGMENTS_WHEN_PAUSED = !!P.enable_segments_when_paused;
    const LIVE_COLOR = P.line_color || "#39FF14";
    const BG = P.bg_color || "#0a0a0a";

    // Audio
    const b64 = "{audio_b64}";
    function b64ToBlob(b64Data) {{
      const byteChars = atob(b64Data);
      const byteNums = new Array(byteChars.length);
      for (let i=0; i<byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
      return new Blob([new Uint8Array(byteNums)], {{ type: "audio/wav" }});
    }}
    const aud = document.getElementById('aud');
    try {{ aud.src = URL.createObjectURL(b64ToBlob(b64)); }} catch(e) {{ console.error(e); }}

    const AC = new (window.AudioContext || window.webkitAudioContext)();
    const src = AC.createMediaElementSource(aud);
    const analyser = AC.createAnalyser();
    analyser.fftSize = FFTSIZE;
    analyser.smoothingTimeConstant = SMOOTH_TC;
    src.connect(analyser); analyser.connect(AC.destination);
    const bins = analyser.frequencyBinCount;
    const freqBytes = new Uint8Array(bins); // FAST PATH

    function binFreq(i) {{ return (i * AC.sampleRate * 0.5) / bins; }}

    // Geometry
    const theta = new Float64Array(phi_grid.length);
    for (let i=0;i<phi_grid.length;i++) theta[i] = 2*Math.PI*phi_grid[i];
    const x = new Float64Array(phi_grid.length);
    const y = new Float64Array(phi_grid.length);
    const xbase = new Float64Array(phi_grid.length);
    const ybase = new Float64Array(phi_grid.length);
    for (let i=0;i<phi_grid.length;i++) {{
      const r = r_base[i];
      xbase[i] = r * Math.cos(theta[i]);
      ybase[i] = r * Math.sin(theta[i]);
    }}

    // Helpers
    function smoothCircular(arr, win) {{
      if (win<=1) return arr.slice();
      const n = arr.length, pad = Math.floor(win/2), out = new Float64Array(n);
      let acc = 0;
      for (let k=-pad; k<=pad; k++) acc += arr[(k+n)%n];
      for (let i=0;i<n;i++) {{
        out[i] = acc / (2*pad+1);
        const outIdx = (i-pad+n)%n, inIdx = (i+pad+1)%n;
        acc += arr[inIdx] - arr[outIdx];
      }}
      return Array.from(out);
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
      return "#" + R.toString(16).padStart(2,"0") + G.toString(16).padStart(2,"0") + B.toString(16).padStart(2,"0");
    }}

    // Plotly traces:
    // 1) Optional colored segments (drawn once on pause)
    const traces = [];
    for (let s=0; s<SPOKES; s++) {{
      const ang = 2*Math.PI * s / SPOKES;
      traces.push({{
        type:"scatter", mode:"lines",
        x:[-Rmax*Math.cos(ang), Rmax*Math.cos(ang)],
        y:[-Rmax*Math.sin(ang), Rmax*Math.sin(ang)],
        line:{{width:1, color:"rgba(255,255,255,0.08)"}}, hoverinfo:"skip", showlegend:false
      }});
    }}
    for (let k=Math.floor(-HALF); k<=Math.ceil(HALF); k++) {{
      const xx=[], yy=[];
      for (let i=0;i<=360;i++) {{
        const a=i*Math.PI/180, r=k+HALF+0.25; xx.push(r*Math.cos(a)); yy.push(r*Math.sin(a));
      }}
      traces.push({{
        type:"scatter", mode:"lines", x:xx, y:yy,
        line:{{width:1, color:"rgba(255,255,255,0.08)"}}, hoverinfo:"skip", showlegend:false
      }});
    }}

    // 2) The single FAST WebGL line (updated every frame) — starts as base
    traces.push({{
      type:"scattergl", mode:"lines",
      x:Array.from(xbase), y:Array.from(ybase),
      line:{{width: LINE_W, color: LIVE_COLOR}},
      hoverinfo:"skip", name:"live"
    }});
    const idxLive = traces.length - 1;

    // 3) An optional segmented static layer (only used when paused, drawn on-demand)
    const idxSegStart = traces.length;
    if (ENABLE_SEGMENTS_WHEN_PAUSED && seg_idx.length>0) {{
      for (const [i,j] of seg_idx) {{
        const mid = 0.5*(phi_grid[i]+phi_grid[j]);
        const hue = ((mid % 1)+1)%1;
        const col = hsvToRgbHex(hue, 0.95, 0.95);
        traces.push({{
          type:"scattergl", mode:"lines",
          x:Array.from(xbase.slice(i, j+1)), y:Array.from(ybase.slice(i, j+1)),
          line:{{width: Math.max(1, LINE_W-1), color: col, opacity: 0.0}}, // hidden initially
          hoverinfo:"skip", showlegend:false
        }});
      }}
    }}

    const layout = {{
      template:"plotly_dark",
      paper_bgcolor:BG, plot_bgcolor:BG,
      xaxis:{{visible:false, range:[-Rmax, Rmax]}},
      yaxis:{{visible:false, range:[-Rmax, Rmax], scaleanchor:"x", scaleratio:1}},
      margin:{{l:10, r:10, t:40, b:10}},
      title:"FAST WebGL Spiral — live line during playback; colored segments when paused"
    }};
    const plotDiv = document.getElementById("plot");
    Plotly.newPlot(plotDiv, traces, layout, {{responsive:true}});

    // Toggle segment visibility when paused/playing (no per-frame updates to segments)
    function showSegments(show) {{
      if (!ENABLE_SEGMENTS_WHEN_PAUSED || seg_idx.length===0) return;
      const idxs = [];
      const vis = show ? 1.0 : 0.0;
      const col = [];
      for (let t=idxSegStart; t<plotDiv.data.length; t++) {{
        idxs.push(t); col.push(vis);
      }}
      // update opacity only
      Plotly.restyle(plotDiv, {{"line.opacity": col}}, idxs);
    }}

    // FAST loop: exactly ONE update per frame, one trace only
    function computeXY(dt) {{
      // Accumulate power into phi bins using fast byte magnitudes
      analyser.getByteFrequencyData(freqBytes); // 0..255
      const energy = new Float64Array(phi_grid.length);
      for (let bi=0; bi<bins; bi++) {{
        const v = freqBytes[bi];
        if (v <= BYTE_FLOOR) continue;
        const f = binFreq(bi);
        if (f < FMIN || f > FMAX) continue;
        const phi = Math.log2(f / A4);
        if (phi < -HALF || phi >= HALF) continue;
        const idx = Math.floor((phi + HALF) / (2*HALF) * phi_grid.length);
        const ii = Math.max(0, Math.min(phi_grid.length - 1, idx));
        // power-ish
        energy[ii] += v * v;
      }}
      // normalize (median/p95)
      const tmp = Array.from(energy).sort((a,b)=>a-b);
      const q = (t)=> tmp[Math.max(0, Math.min(tmp.length-1, Math.floor(t*(tmp.length-1))))];
      const med = q(0.5), p95 = q(0.95);
      let scale = p95 - med; if (scale <= 1e-9) scale = 1e-9;
      for (let i=0;i<energy.length;i++) energy[i] = (energy[i] - med) / scale;

      // temporal LPF
      const alpha = 1 - Math.exp(-2*Math.PI*RIP_FC*dt);
      if (!window._lp || window._lp.length !== energy.length) window._lp = new Float64Array(energy.length);
      const lp = window._lp;
      for (let i=0;i<energy.length;i++) lp[i] += alpha * (energy[i] - lp[i]);

      // gamma
      if (Math.abs(RIP_GAM - 1.0) > 1e-6) {{
        for (let i=0;i<lp.length;i++) {{
          const s = Math.sign(lp[i]), a = Math.abs(lp[i]); lp[i] = s * Math.pow(a, RIP_GAM);
        }}
      }}

      // spatial smoothing (apply once; trail is disabled in fast path)
      const sm = (SMOOTH>1) ? smoothCircular(Array.from(lp), SMOOTH) : Array.from(lp);

      // zero-center + auto-gain
      let mean = 0; for (let i=0;i<sm.length;i++) mean += sm[i]; mean/=sm.length;
      for (let i=0;i<sm.length;i++) sm[i] -= mean;

      let rms = 0; for (let i=0;i<sm.length;i++) rms += sm[i]*sm[i];
      rms = Math.sqrt(rms / sm.length);
      let gain = (rms>1e-9) ? (RIP_TGT / rms) : 0.0;
      gain *= Math.max(0.0, WGAIN);
      const RCLAMP = Math.max(0.01, RIP_MAX);

      for (let i=0;i<phi_grid.length;i++) {{
        const dr = Math.max(-RCLAMP, Math.min(RCLAMP, gain * sm[i]));
        const r = r_base[i] + dr;
        x[i] = r * Math.cos(theta[i]);
        y[i] = r * Math.sin(theta[i]);
      }}
    }}

    let lastTS = 0;
    function loop(ts) {{
      const dt = lastTS ? (ts - lastTS)/1000 : 1/60; lastTS = ts;
      if (!aud.paused) {{
        computeXY(dt);
        // single update: live line only
        Plotly.restyle(plotDiv, {{
          x: [Array.from(x)],
          y: [Array.from(y)],
          "line.color": [[LIVE_COLOR]],
          "line.width": [[LINE_W]]
        }}, [plotDiv.data.length-1]); // idxLive
      }}
      requestAnimationFrame(loop);
    }}
    requestAnimationFrame(loop);

    // controls
    document.getElementById("play").addEventListener("click", async () => {{
      try {{
        if (AC.state === "suspended") await AC.resume();
        aud.play();
        showSegments(false); // hide segmented while playing
      }} catch(e) {{ console.error(e); }}
    }});
    document.getElementById("pause").addEventListener("click", () => {{
  aud.pause();
  showSegments(true);  // show pretty segmented spiral when paused

  // Snap the live WebGL line back to the base spiral so the paused view is clean.
  // Use the correct JS trace index we defined earlier: idxLive.
  Plotly.restyle(plotDiv, {{
    x: [Array.from(xbase)],
    y: [Array.from(ybase)]
  }}, [idxLive]);
}});</script>
</body>
</html>
""", height=720)

