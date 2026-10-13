from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .calibration import PilotCalibrator


class SpectralEngine:
    """Embed and extract a binary payload using pairwise magnitude modulation in the 2D FFT domain."""

    def __init__(self, block_size: int = 64, r_min: float = 8.0, r_max: float = 22.0, seed: int = 2026) -> None:
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("block_size must be a positive even integer")
        self.block_size = int(block_size)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.seed = int(seed)
        self.calibrator = PilotCalibrator(block_size=self.block_size)
        # Lazy cache of vectorized (row, col) coordinate arrays per bit count.
        self._bin_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_bin_pairs(self, num_bits: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Return deterministic non-overlapping FFT coordinate pairs for the requested bit count."""
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")

        candidates: list[tuple[int, int]] = []
        v_limit = self.block_size // 2 - 1
        pilot_coords = set(self.calibrator.get_pilot_coords())
        for u in range(self.block_size):
            for v in range(1, v_limit):
                if (u, v) in pilot_coords:
                    continue
                u_freq = min(u, self.block_size - u)
                radius = math.hypot(float(u_freq), float(v))
                if self.r_min <= radius <= self.r_max:
                    candidates.append((u, v))

        required = 2 * num_bits
        if len(candidates) < required:
            raise ValueError(
                f"Not enough valid FFT bin pairs: need {required}, have {len(candidates)}. "
                "Adjust block_size, r_min, or r_max."
            )

        rng = np.random.RandomState(self.seed)
        selection = rng.choice(len(candidates), size=required, replace=False)
        selected = [candidates[int(idx)] for idx in selection]
        return selected[:num_bits], selected[num_bits:]

    def get_bin_arrays(self, num_bits: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized (row, col) coordinate arrays for the given bit count (cached)."""
        cached = self._bin_arrays.get(num_bits)
        if cached is None:
            pairs_a, pairs_b = self.get_bin_pairs(num_bits)
            a = np.asarray(pairs_a, dtype=np.intp)
            b = np.asarray(pairs_b, dtype=np.intp)
            cached = (a, b)
            self._bin_arrays[num_bits] = cached
        return cached

    def embed_tile(
        self,
        tile: np.ndarray,
        bits: np.ndarray,
        pairs_a: list[tuple[int, int]],
        pairs_b: list[tuple[int, int]],
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Embed bits into a single tile while preserving the original spectrum phase."""
        tile_array = np.asarray(tile, dtype=np.float64)
        if tile_array.shape != (self.block_size, self.block_size):
            raise ValueError(f"Tile shape must be ({self.block_size}, {self.block_size})")

        bit_array = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if len(bit_array) != len(pairs_a) or len(pairs_a) != len(pairs_b):
            raise ValueError("Bit length and pair count must match")

        spectrum = np.fft.rfft2(tile_array)

        for bit, (coord_a, coord_b) in zip(bit_array, zip(pairs_a, pairs_b)):
            (u_a, v_a) = coord_a
            (u_b, v_b) = coord_b

            x_a = spectrum[u_a, v_a]
            x_b = spectrum[u_b, v_b]
            mu = (abs(x_a) + abs(x_b)) / 2.0
            delta = max(float(alpha) * mu, 1.5)

            if bit == 1:
                target_a = mu + delta / 2.0
                target_b = max(mu - delta / 2.0, 0.05)
            else:
                target_a = max(mu - delta / 2.0, 0.05)
                target_b = mu + delta / 2.0

            spectrum[u_a, v_a] = target_a * np.exp(1j * np.angle(x_a))
            spectrum[u_b, v_b] = target_b * np.exp(1j * np.angle(x_b))

        magnitudes = np.abs(spectrum)
        self.calibrator.inject_pilots(spectrum, float(np.mean(magnitudes)))
        spatial = np.fft.irfft2(spectrum, s=(self.block_size, self.block_size))
        return np.clip(spatial, 0.0, 255.0).astype(np.float64)

    def extract_bits(
        self,
        tile: np.ndarray,
        pairs_a: list[tuple[int, int]],
        pairs_b: list[tuple[int, int]],
    ) -> np.ndarray:
        """Blindly extract embedded bits based on pairwise magnitude comparison."""
        tile_array = np.asarray(tile, dtype=np.float64)
        if tile_array.shape != (self.block_size, self.block_size):
            raise ValueError(f"Tile shape must be ({self.block_size}, {self.block_size})")

        spectrum = np.fft.rfft2(tile_array)
        return self._compare_pairs(spectrum, pairs_a, pairs_b)

    @staticmethod
    def _pairs_to_arrays(
        pairs_a: Sequence[tuple[int, int] | np.ndarray] | np.ndarray,
        pairs_b: Sequence[tuple[int, int] | np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coerce pair sequences into (row, col) intp arrays; 2-D arrays pass through."""
        a = np.asarray(pairs_a, dtype=np.intp)
        b = np.asarray(pairs_b, dtype=np.intp)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("pairs_a must be a sequence of (row, col) coordinates")
        return a, b

    def _compare_pairs(self, spectrum: np.ndarray, pairs_a, pairs_b) -> np.ndarray:
        """Vectorized pairwise magnitude comparison of one spectrum."""
        a, b = self._pairs_to_arrays(pairs_a, pairs_b)
        mag = np.abs(spectrum)
        return (mag[a[:, 0], a[:, 1]] > mag[b[:, 0], b[:, 1]]).astype(np.uint8)

    @staticmethod
    def _sync_bits(sync_marker: bytes) -> np.ndarray:
        return np.unpackbits(np.frombuffer(sync_marker, dtype=np.uint8), bitorder="big")

    def extract_bits_from_spectrum(
        self,
        spectrum: np.ndarray,
        pairs_a: list[tuple[int, int]],
        pairs_b: list[tuple[int, int]],
    ) -> np.ndarray:
        """Extracts bits from an existing 2D rFFT spectrum via pairwise magnitude comparison."""
        return self._compare_pairs(spectrum, pairs_a, pairs_b)

    def check_sync_preamble(
        self,
        spectrum: np.ndarray,
        pairs_a: list[tuple[int, int]],
        pairs_b: list[tuple[int, int]],
        sync_marker: bytes,
    ) -> bool:
        """O(1)-ish fast-path: test whether the first 16 spectrum bits match the sync preamble."""
        probed = self._compare_pairs(spectrum, pairs_a[:16], pairs_b[:16])
        return bool(np.array_equal(probed, self._sync_bits(sync_marker)))

    def sync_probe(self, spectrum: np.ndarray, a: np.ndarray, b: np.ndarray, sync: bytes) -> bool:
        """Vectorized sync check using precomputed 16-coordinate (row, col) arrays."""
        probed = self._compare_pairs(spectrum, a[:16], b[:16])
        return np.packbits(probed, bitorder="big").tobytes() == sync
