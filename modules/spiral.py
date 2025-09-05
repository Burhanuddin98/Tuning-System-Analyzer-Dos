# modules/spiral.py
from __future__ import annotations
import numpy as np
from modules.config import SpiralUI, RippleUI, AnalyserUI

def build_spiral_params(a4_ref: float, ui: SpiralUI) -> dict:
    turns = int(ui.turns)
    bins_total = int(ui.bins_per_turn * turns)
    half = turns / 2.0
    phi = np.linspace(-half, +half, bins_total, endpoint=False)
    r_base = phi + half + 0.25
    theta = 2.0*np.pi*phi
    x0 = r_base*np.cos(theta); y0 = r_base*np.sin(theta)
    Rmax = float(np.max(np.sqrt(x0**2 + y0**2)) + 0.75)

    # segment breaks for colored line (stable across frames)
    chunk = max(1, min(len(phi)//2, int(np.ceil(len(phi) / ui.max_segments))))
    seg_idx = []
    i = 0
    while i < len(phi)-1:
        j = min(len(phi)-1, i + chunk)
        seg_idx.append((i, j))
        i = j

    return dict(
        # geometry
        phi_grid=phi.tolist(),
        r_base=r_base.tolist(),
        seg_idx=seg_idx,
        Rmax=Rmax,
        half=half,
        spokes=int(ui.spokes),
        line_w=int(ui.line_width),
        # reference
        a4_ref=float(a4_ref),
    )

def render_html_realtime(spiral_params: dict, ripple_ui: RippleUI, analyser_ui: AnalyserUI) -> str:
    """
    Returns a self-contained HTML string that:
      - reconstructs WAV from base64 (injected later),
      - uses WebAudio Analyser to compute live energy,
      - renders a colored, rippling Archimedean spiral with Plotly.
    """
    return f"""
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
    const P = {{...JSON.parse(`{{PARAMS_JSON}}`)}};

    // ---- Rebuild WAV Blob from base64 ----
    const b64 = "{{AUDIO_B64}}";
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

    // ---- WebAudio ----
    const AC = new (window.AudioContext || window.webkitAudioContext)();
    const src = AC.createMediaElementSource(aud);
    const analyser = AC.createAnalyser();
    analyser.fftSize = P.fft_size;
    // cap at 0.5 because we do our own temporal LPF for the "cardiogram" feel
    analyser.smoothingTimeConstant = Math.min(0.5, Math.max(0.0, P.smoothing_tc));
    src.connect(analyser); analyser.connect(AC.destination);

    const bins = analyser.frequencyBinCount;
    const freqData = new Float32Array(bins);
    function binFreq(i) {{ return (i * AC.sampleRate * 0.5) / bins; }}

    // ---- Plotly init (guides + colored segment lines) ----
    const phi_grid = Float64Array.from(P.phi_grid);
    const r_base   = Float64Array.from(P.r_base);
    const seg_idx  = P.seg_idx;
    const HALF = P.half, SPOKES = P.spokes, LINE_W = P.line_w;
    const Rmax = P.Rmax, A4 = P.a4_ref;

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

    // precompute theta
    const theta = new Float64Array(phi_grid.length);
    for (let i=0;i<phi_grid.length;i++) theta[i] = 2*Math.PI*phi_grid[i];

    // colored segments (line itself is colored)
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
        x: new Array(j-i+1).fill(0),
        y: new Array(j-i+1).fill(0),
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
      title:"TRUE Spiral â€” angle: pitch class â€¢ radius: octaves (colored; LIVE cardiogram ripples)",
    }};
    const plotDiv = document.getElementById('plot');
    Plotly.newPlot(plotDiv, traces, layout);

    // circular moving average
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

    const x = new Float64Array(phi_grid.length);
    const y = new Float64Array(phi_grid.length);
    let lastTS = 0;

    function tick(ts) {{
      if (!lastTS) lastTS = ts;
      const dt = (ts - lastTS) / 1000.0;
      const minDt = 1.0 / Math.max(10, P.target_fps);
      if (dt < minDt) {{ requestAnimationFrame(tick); return; }}
      lastTS = ts;

      if (!aud.paused) {{
        analyser.getFloatFrequencyData(freqData); // dBFS negatives

        // --- ACCUMULATE POWER onto phi bins ---
        const energy = new Float64Array(phi_grid.length);
        for (let bi = 0; bi < bins; bi++) {{
          const db = freqData[bi];
          if (!isFinite(db) || db < P.peak_db) continue;
          const f = binFreq(bi);
          if (f < P.fmin || f > P.fmax) continue;
          const phi = Math.log2(f / A4);
          if (phi < -HALF || phi >= HALF) continue;
          const idx = Math.floor((phi + HALF) / (2 * HALF) * phi_grid.length);
          const ii = Math.max(0, Math.min(phi_grid.length - 1, idx));
          const pwr = Math.pow(10, db / 10);   // POWER domain (stronger)
          energy[ii] += pwr;
        }}

        // --- Robust normalize (median/p95) ---
        const tmp = Array.from(energy).sort((a,b)=>a-b);
        const q = (t)=> tmp[Math.max(0, Math.min(tmp.length-1, Math.floor(t*(tmp.length-1))))];
        const med = q(0.5), p95 = q(0.95);
        let scale = p95 - med; if (scale <= 1e-12) scale = 1e-12;
        for (let i=0;i<energy.length;i++) energy[i] = (energy[i] - med) / scale;

        // --- Temporal LPF (cardiogram feel) ---
        const alpha = 1 - Math.exp(-2*Math.PI*Math.max(0.01, P.ripple_temporal_cutoff)*dt);
        if (!window._energyLP || window._energyLP.length !== energy.length) {{
          window._energyLP = new Float64Array(energy.length);
        }}
        const energyLP = window._energyLP;
        for (let i=0;i<energy.length;i++) {{
          energyLP[i] += alpha * (energy[i] - energyLP[i]);
        }}

        // --- Gamma (contrast) ---
        if (Math.abs(P.ripple_gamma - 1.0) > 1e-6) {{
          for (let i=0;i<energyLP.length;i++) {{
            const s = Math.sign(energyLP[i]), a = Math.abs(energyLP[i]);
            energyLP[i] = s * Math.pow(a, P.ripple_gamma);
          }}
        }}

        // --- Double spatial smoothing (â‰ˆ Gaussian) ---
        const sm1 = (P.smooth_bins > 1) ? smoothCircular(Array.from(energyLP), P.smooth_bins) : Array.from(energyLP);
        const energySm = (P.smooth_bins > 1) ? smoothCircular(sm1, P.smooth_bins) : sm1;

        // --- Zero-center, auto-gain, clamp ---
        let mean = 0; for (let i=0;i<energySm.length;i++) mean += energySm[i];
        mean /= energySm.length;
        for (let i=0;i<energySm.length;i++) energySm[i] -= mean;

        let rms = 0; for (let i=0;i<energySm.length;i++) rms += energySm[i]*energySm[i];
        rms = Math.sqrt(rms / energySm.length);
        let gain = (rms > 1e-9) ? (P.ripple_target / rms) : 0.0;
        gain *= Math.max(0.0, P.wiggle_gain);
        const RCLAMP = Math.max(0.01, P.ripple_max);

        for (let i=0;i<phi_grid.length;i++) {{
          const dr = Math.max(-RCLAMP, Math.min(RCLAMP, gain * energySm[i]));
          const r = r_base[i] + dr;
          x[i] = r * Math.cos(theta[i]);
          y[i] = r * Math.sin(theta[i]);
        }}

        // Update only the colored segments
        const updateX = [], updateY = [], idxs = [];
        for (let s=0; s<seg_idx.length; s++) {{
          const [i0, j0] = seg_idx[s];
          updateX.push(Array.from(x.slice(i0, j0+1)));
          updateY.push(Array.from(y.slice(i0, j0+1)));
          idxs.push({{ 'trace': (segStart + s) }});
        }}
        // Plotly.update signature: (gd, data, layout, traces)
        Plotly.update(plotDiv, {{x: updateX, y: updateY}}, {{}}, idxs.map(o => o.trace));
      }}

      requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);

    // Buttons
    document.getElementById('play').addEventListener('click', async () => {{
      try {{ if (AC.state === 'suspended') await AC.resume(); aud.play(); }} catch(e){{ console.error(e); }}
    }});
    document.getElementById('pause').addEventListener('click', () => aud.pause());
  </script>
</body>
</html>
"""


