from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


class SpectralEngine:
    """Embed and extract a binary payload using pairwise magnitude modulation in the 2D FFT domain."""

    def __init__(self, block_size: int = 64, r_min: float = 8.0, r_max: float = 22.0, seed: int = 2026) -> None:
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("block_size must be a positive even integer")
        self.block_size = int(block_size)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.seed = int(seed)

    def get_bin_pairs(self, num_bits: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Return deterministic non-overlapping FFT coordinate pairs for the requested bit count."""
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")

        candidates: List[Tuple[int, int]] = []
        v_limit = self.block_size // 2 - 1
        for u in range(self.block_size):
            for v in range(1, v_limit):
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

    def embed_tile(
        self,
        tile: np.ndarray,
        bits: np.ndarray,
        pairs_a: List[Tuple[int, int]],
        pairs_b: List[Tuple[int, int]],
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

        spatial = np.fft.irfft2(spectrum, s=(self.block_size, self.block_size))
        return np.clip(spatial, 0.0, 255.0).astype(np.float64)

    def extract_bits(
        self,
        tile: np.ndarray,
        pairs_a: List[Tuple[int, int]],
        pairs_b: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Blindly extract embedded bits based on pairwise magnitude comparison."""
        tile_array = np.asarray(tile, dtype=np.float64)
        if tile_array.shape != (self.block_size, self.block_size):
            raise ValueError(f"Tile shape must be ({self.block_size}, {self.block_size})")

        spectrum = np.fft.rfft2(tile_array)
        extracted = np.empty(len(pairs_a), dtype=np.uint8)

        for idx, (coord_a, coord_b) in enumerate(zip(pairs_a, pairs_b)):
            (u_a, v_a) = coord_a
            (u_b, v_b) = coord_b
            extracted[idx] = 1 if abs(spectrum[u_a, v_a]) > abs(spectrum[u_b, v_b]) else 0

        return extracted
