"""Step 1: read the tilt of a page modulo 90 degrees and rotate it level.

The tilt estimator is the radial projection of jdeskew (Pham et al., ICIP 2022) with two corrections:

* the binarised page is multiplied by a 2-D Hann window before the FFT. Like jdeskew, we pad the page to a square with
  white and then threshold it, so a photograph (darker than white) gains a solid ink line where it meets the padding;
  that straight edge puts a bright cross on the frequency axes and the projection peaks at 0 degrees. The taper fades
  the page boundary and the line with it. (Thresholding before padding removes the line too.)
* a peak at the edge of the search range is returned as is. The released code maps it to 0 degrees, a rule meant
  for blank pages; here 0 is returned only when the projection is flat.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def estimate_tilt(image: Image.Image | np.ndarray, height: int = 2048, angle_max: float = 45.0, steps_per_degree: int = 20) -> float:
    """Clockwise tilt of the text lines in degrees, in [-angle_max, angle_max]; rotating the page
    counterclockwise by this angle levels it."""
    a = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    a = cv2.resize(a, None, fx=height / a.shape[0], fy=height / a.shape[0])
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a   # as jdeskew does on an RGB array; keeps the paper's numbers
    n = cv2.getOptimalDFTSize(max(gray.shape))
    square = cv2.copyMakeBorder(gray, 0, n - gray.shape[0], 0, n - gray.shape[1], cv2.BORDER_CONSTANT, value=255)
    ink = cv2.adaptiveThreshold(~square, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10).astype(np.float64)
    window = np.zeros_like(ink)
    window[: gray.shape[0], : gray.shape[1]] = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))   # taper the page, not the padding
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(ink * window)))
    c = spectrum.shape[0] // 2
    r = np.arange(c)
    thetas = np.linspace(-angle_max, angle_max, int(angle_max * steps_per_degree * 2)) / 180 * np.pi
    projection = np.array([spectrum[c + (r * np.cos(t)).astype(np.int32), c + (-r * np.sin(t)).astype(np.int32)].sum() for t in thetas])
    if projection.max() == projection.min():   # a blank page has no tilt to read
        return 0.0
    return float(thetas[int(np.argmax(projection))] / np.pi * 180)


def level(image: Image.Image, tilt: float, snap: float = 1.0, crop_to_ink: bool = True) -> Image.Image:
    """Rotate the page counterclockwise by `tilt`. Corners take the median colour of the page border, so a photograph
    gains no white frame; the result is cropped to its ink. Tilts below `snap` degrees are left alone."""
    if abs(tilt) < snap:   # a level page passes through untouched
        return image
    pixels = np.asarray(image.convert("RGB"))
    edge = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    fill = tuple(int(v) for v in np.median(edge, 0))
    out = image.convert("RGB").rotate(tilt, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
    if not crop_to_ink:
        return out
    ink = np.asarray(out.convert("L")) < 128
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    return out if not rows.size else out.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))
