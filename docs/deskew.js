// LevelHead Step 1 in the browser: jdeskew's radial projection with the two corrections (Hann taper, keep edge peak).
// gray: Float32Array of luminance 0..255, w x h, already scaled so max(w, h) <= N. Returns tilt (clockwise degrees)
// plus the spectrum and projection for display. Mirrors levelhead/deskew.py.
(function (root) {
  function fft1(re, im, n, inv) {               // in-place iterative radix-2 FFT
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inv ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2, xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }
  function fft2(re, im, n) {
    const r = new Float64Array(n), i = new Float64Array(n);
    for (let y = 0; y < n; y++) { const o = y * n; for (let x = 0; x < n; x++) { r[x] = re[o + x]; i[x] = im[o + x]; } fft1(r, i, n); for (let x = 0; x < n; x++) { re[o + x] = r[x]; im[o + x] = i[x]; } }
    for (let x = 0; x < n; x++) { for (let y = 0; y < n; y++) { r[y] = re[y * n + x]; i[y] = im[y * n + x]; } fft1(r, i, n); for (let y = 0; y < n; y++) { re[y * n + x] = r[y]; im[y * n + x] = i[y]; } }
  }
  function blur(src, w, h) {                    // OpenCV's Gaussian for a 15 x 15 window: sigma = 2.6, replicated borders
    const k = [], s = 2.6; let sum = 0; for (let i = -7; i <= 7; i++) { const v = Math.exp(-i * i / (2 * s * s)); k.push(v); sum += v; } for (let i = 0; i < 15; i++) k[i] /= sum;
    const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let a = 0; for (let i = -7; i <= 7; i++) a += k[i + 7] * src[y * w + Math.min(w - 1, Math.max(0, x + i))]; tmp[y * w + x] = a; }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let a = 0; for (let i = -7; i <= 7; i++) a += k[i + 7] * tmp[Math.min(h - 1, Math.max(0, y + i)) * w + x]; out[y * w + x] = a; }
    return out;
  }
  function estimate(gray, w, h, opts) {
    const N = opts.N, hann = opts.hann !== false, keepEdge = opts.keepEdge !== false, amax = 45, steps = 1800;
    // As jdeskew: pad the page to N x N with white (no ink after inversion), THEN threshold. Along the boundary between a
    // page darker than white and the padding, the threshold draws a solid ink line; the Hann taper fades it out.
    const inv = new Float32Array(N * N); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) inv[y * N + x] = 255 - gray[y * w + x];
    const mean = blur(inv, N, N);
    const re = new Float64Array(N * N), im = new Float64Array(N * N);
    const wx = new Float64Array(w), wy = new Float64Array(h);
    for (let x = 0; x < w; x++) wx[x] = hann ? 0.5 - 0.5 * Math.cos(2 * Math.PI * x / (w - 1)) : 1;
    for (let y = 0; y < h; y++) wy[y] = hann ? 0.5 - 0.5 * Math.cos(2 * Math.PI * y / (h - 1)) : 1;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x;
      if (inv[i] > mean[i] + 10) re[i] = 255 * (y < h && x < w ? wy[y] * wx[x] : (hann ? 0 : 1));   // the taper covers the page, padding gets 0
    }
    fft2(re, im, N);
    const mag = new Float32Array(N * N), c = N >> 1;                     // fftshifted magnitude
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const i = y * N + x; mag[((y + c) % N) * N + ((x + c) % N)] = Math.hypot(re[i], im[i]); }
    const proj = new Float64Array(steps), angles = new Float64Array(steps);
    for (let s = 0; s < steps; s++) {
      const t = (-amax + 2 * amax * s / (steps - 1)) / 180 * Math.PI, ct = Math.cos(t), st = Math.sin(t); angles[s] = t * 180 / Math.PI;
      let a = 0; for (let r = 0; r < c; r++) a += mag[(c + Math.trunc(r * ct)) * N + (c + Math.trunc(-r * st))];
      proj[s] = a;
    }
    let best = 0, lo = Infinity, hi = -Infinity; for (let s = 0; s < steps; s++) { if (proj[s] > proj[best]) best = s; lo = Math.min(lo, proj[s]); hi = Math.max(hi, proj[s]); }
    let tilt = angles[best];
    if (hi === lo) tilt = 0;                                             // blank page
    else if (!keepEdge && best === 0) tilt = 0;                          // the released code's edge rule
    return { tilt, mag, N, proj, angles, best };
  }
  root.LHDeskew = { estimate };
  if (typeof module !== "undefined") module.exports = root.LHDeskew;
})(typeof window !== "undefined" ? window : globalThis);
