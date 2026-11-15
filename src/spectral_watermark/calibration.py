from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import maximum_filter  # type: ignore[import-untyped]

PILOT_REFERENCE_R = 16.0
PILOT_BOOST_FACTOR = 4.5


class PilotCalibrator:
    """Inject and detect spectral pilot peaks used to estimate display scaling."""

    def __init__(self, block_size: int = 64, r_ref: float = PILOT_REFERENCE_R) -> None:
        if block_size <= 0 or block_size % 2:
            raise ValueError("block_size must be a positive even integer")
        if not np.isfinite(r_ref) or r_ref <= 0:
            raise ValueError("r_ref must be finite and positive")
        self.block_size = block_size
        self.r_ref = float(r_ref)
        self.coord_offset = round(self.r_ref / math.sqrt(2.0))

    def get_pilot_coords(self) -> list[tuple[int, int]]:
        """Return the two stored rFFT coordinates for the symmetric pilot pair."""
        return [
            (self.coord_offset, self.coord_offset),
            (self.block_size - self.coord_offset, self.coord_offset),
        ]

    def inject_pilots(self, tile_spectrum: np.ndarray, mean_magnitude: float) -> np.ndarray:
        """Boost calibration-bin magnitudes while retaining their existing phases."""
        boosted_mag = max(float(mean_magnitude) * PILOT_BOOST_FACTOR, 30.0)
        for u, v in self.get_pilot_coords():
            phase = np.angle(tile_spectrum[u, v])
            tile_spectrum[u, v] = boosted_mag * np.exp(1j * phase)
        return tile_spectrum

    def estimate_scale(
        self,
        cropped_image: np.ndarray,
        search_radius_min: float = 8.0,
        search_radius_max: float = 28.0,
    ) -> float:
        """Estimate apparent display scale from the strongest full-canvas FFT peaks."""
        img = np.asarray(cropped_image)
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            else:
                img = img[:, :, :3] @ np.array([0.299, 0.587, 0.114])
        if img.ndim != 2:
            raise ValueError("Image must be 2D grayscale or an RGB(A) image")
        img = img.astype(np.float64, copy=False)
        height, width = img.shape
        if height < self.block_size or width < self.block_size:
            return 1.0

        window = np.outer(np.hanning(height), np.hanning(width))
        spectrum = np.abs(np.fft.rfft2((img - np.mean(img)) * window))
        local_max = spectrum == maximum_filter(spectrum, size=7)
        threshold = np.percentile(spectrum, 99.2)
        peak_rows, peak_cols = np.nonzero(local_max & (spectrum > threshold))
        candidate_peaks: list[tuple[float, float, int, int]] = []
        for row, col in zip(peak_rows, peak_cols):
            if col == 0:
                continue
            norm_row = min(row, height - row) * self.block_size / height
            norm_col = col * self.block_size / width
            radius = math.hypot(norm_row, norm_col)
            if search_radius_min <= radius <= search_radius_max:
                candidate_peaks.append((float(spectrum[row, col]), radius, int(row), int(col)))

        if not candidate_peaks:
            return 1.0

        pilot_pairs = [
            (first, second)
            for index, first in enumerate(candidate_peaks)
            for second in candidate_peaks[index + 1 :]
            if abs(first[2] + second[2] - (height - 1)) <= 3
            and abs(first[3] - second[3]) <= 2
        ]
        if not pilot_pairs:
            return 1.0
        strongest_pair = max(pilot_pairs, key=lambda pair: min(pair[0][0], pair[1][0]))
        detected_radius = float(np.median([strongest_pair[0][1], strongest_pair[1][1]]))
        reference_radius = math.sqrt(2.0) * self.coord_offset
        estimated_scale = reference_radius / detected_radius
        return estimated_scale if 0.5 <= estimated_scale <= 2.5 else 1.0


def trim_solid_borders(image: np.ndarray, variance_threshold: float = 12.0) -> np.ndarray:
    """Strip low-variance outer rows and columns from screenshots when safe."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        variance_image = arr.astype(np.float64, copy=False)
    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        variance_image = arr[:, :, :3].astype(np.float64, copy=False)
        if variance_image.shape[2] == 1:
            variance_image = variance_image[:, :, 0]
    else:
        raise ValueError("Image must be 2D grayscale or an RGB(A) image")

    if variance_image.ndim == 2:
        row_variance = np.var(variance_image, axis=1)
        col_variance = np.var(variance_image, axis=0)
    else:
        row_variance = np.mean(np.var(variance_image, axis=1), axis=1)
        col_variance = np.mean(np.var(variance_image, axis=0), axis=1)
    valid_rows = np.flatnonzero(row_variance > variance_threshold)
    valid_cols = np.flatnonzero(col_variance > variance_threshold)
    if valid_rows.size <= 32 or valid_cols.size <= 32:
        return arr
    return arr[valid_rows[0] : valid_rows[-1] + 1, valid_cols[0] : valid_cols[-1] + 1]
