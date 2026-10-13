from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from PIL import Image

from .calibration import trim_solid_borders
from .core import SpectralEngine
from .ecc import BitCodec


class WatermarkPipeline:
    """End-to-end embedding and extraction pipeline for a crop-resilient FFT watermark."""

    def __init__(
        self,
        block_size: int = 64,
        r_min: float = 8.0,
        r_max: float = 22.0,
        seed: int = 2026,
    ) -> None:
        self.codec = BitCodec()
        self.engine = SpectralEngine(block_size=block_size, r_min=r_min, r_max=r_max, seed=seed)
        self.calibrator = self.engine.calibrator

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            return arr.astype(np.float64)
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            if arr.shape[2] == 1:
                return arr[:, :, 0].astype(np.float64)
            rgb = arr[:, :, :3]
            luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
            return luminance.astype(np.float64)
        raise ValueError("Image must be 2D grayscale or an RGB(A) image")

    def embed_image(self, image: np.ndarray, payload: str, alpha: float = 0.45) -> tuple[np.ndarray, int]:
        """Embed one payload into every tile of the image and return the stego image and bit length."""
        grayscale = self._to_grayscale(image)
        bits = self.codec.encode_payload(payload)
        pairs_a, pairs_b = self.engine.get_bin_pairs(len(bits))

        height, width = grayscale.shape
        stego = grayscale.copy()
        block_size = self.engine.block_size

        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                tile = stego[y : y + block_size, x : x + block_size]
                tile_pad = np.zeros((block_size, block_size), dtype=np.float64)
                tile_pad[: tile.shape[0], : tile.shape[1]] = tile
                embedded_block = self.engine.embed_tile(tile_pad, bits, pairs_a, pairs_b, alpha=alpha)
                stego[y : y + tile.shape[0], x : x + tile.shape[1]] = embedded_block[: tile.shape[0], : tile.shape[1]]

        return np.clip(stego, 0.0, 255.0).astype(np.uint8), len(bits)

    def extract_from_crop(
        self,
        cropped_image: np.ndarray,
        expected_bit_len: int | None = None,
        search_stride: int = 2,
        scales: list[float] | None = None,
        min_chars: int = 1,
        max_chars: int = 32,
        on_progress: Callable[[int, int, int, int], None] | None = None,
        timeout: float | None = None,
        on_scale_detected: Callable[[float], None] | None = None,
    ) -> str | None:
        """Two-tier hierarchical search for an embedded watermark inside a crop or screenshot.

        Tier 1 scans block-aligned candidate windows first (fast path for uncropped or
        block-aligned screenshots). Tier 2 falls back to a strided sliding scan over all
        remaining candidate offsets. Each candidate window's 2D rFFT spectrum is computed
        exactly once and reused across all candidate payload lengths; a vectorized 16-bit
        sync preamble check rejects the vast majority of windows before Reed-Solomon
        decoding is ever attempted.

        ``on_progress`` is called as ``on_progress(tier, checked, total, y, x)`` after every
        window so callers can render progress. ``timeout`` (seconds, wall clock) aborts
        the scan early and returns ``None``; ``None`` disables the limit.
        """
        if search_stride <= 0:
            raise ValueError("search_stride must be positive")

        block_size = self.engine.block_size
        trimmed_image = trim_solid_borders(cropped_image)
        base_image = self._to_grayscale(trimmed_image)
        scales_to_test = [] if scales is None else scales
        if any(not np.isfinite(scale) or scale <= 0 for scale in scales_to_test):
            raise ValueError("scales must contain only finite positive values")
        detected_scale = self.calibrator.estimate_scale(base_image)
        if on_scale_detected is not None:
            on_scale_detected(detected_scale)

        # Candidate payload lengths: explicit bit length, or a dynamic character-length scan.
        if expected_bit_len is not None:
            bit_lens: list[int] = [int(expected_bit_len)]
        else:
            bit_lens = [self._payload_bit_len(n) for n in range(int(min_chars), int(max_chars) + 1)]

        # Vectorized coordinate arrays per candidate length; infeasible lengths are skipped.
        per_length: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for bit_len in bit_lens:
            try:
                per_length[bit_len] = self.engine.get_bin_arrays(bit_len)
            except ValueError:
                # Not enough spectral bins for this length; no valid embedding could exist.
                continue
        if not per_length:
            return None

        sync = self.codec.sync_marker
        max_sync = len(sync) * 8
        deadline = time.perf_counter() + timeout if timeout is not None else None
        checked = 0

        def report(tier: int, total: int, y: int, x: int) -> None:
            nonlocal checked
            checked += 1
            if on_progress is not None:
                on_progress(tier, checked, total, y, x)

        def timed_out() -> bool:
            return deadline is not None and time.perf_counter() > deadline

        scans: list[tuple[np.ndarray, float]] = []
        if abs(detected_scale - 1.0) > 0.02:
            scans.append((base_image, detected_scale))
        for scale in scales_to_test:
            if not any(abs(scale - existing_scale) <= 1e-3 for _, existing_scale in scans):
                scans.append((base_image, scale))
        if not any(abs(1.0 - existing_scale) <= 1e-3 for _, existing_scale in scans):
            scans.append((base_image, 1.0))

        for image, apparent_scale in scans:
            height, width = image.shape
            window_size = max(round(block_size * apparent_scale), 1)
            if height < window_size or width < window_size:
                continue
            last_y = height - window_size
            last_x = width - window_size

            def scan_window(
                y: int,
                x: int,
                image: np.ndarray = image,
                window_size: int = window_size,
            ) -> str | None:
                window = image[y : y + window_size, x : x + window_size].astype(np.float64)
                if window_size != block_size:
                    window = np.asarray(
                        Image.fromarray(window).resize(
                            (block_size, block_size), Image.Resampling.BICUBIC
                        ),
                        dtype=np.float64,
                    )
                spectrum = np.fft.rfft2(window)
                # Sync preamble fast-path (vectorized per length): reject without RS decoding.
                for bit_len, (a, b) in per_length.items():
                    if not self.engine.sync_probe(spectrum, a, b, sync):
                        continue
                    if bit_len <= max_sync:
                        return None
                    bits = self.engine._compare_pairs(spectrum, a, b)
                    decoded = self.codec.decode_packet(bits)
                    if decoded is not None:
                        return decoded
                return None

            # Tier 1: block-aligned grid scan.
            tier1_total = (last_y // window_size + 1) * (last_x // window_size + 1)
            for y in range(0, height - window_size + 1, window_size):
                for x in range(0, width - window_size + 1, window_size):
                    report(1, tier1_total, y, x)
                    result = scan_window(y, x)
                    if result is not None:
                        return result
                    if timed_out():
                        return None

            # Tier 2: strided sliding scan over all remaining candidate offsets.
            tier2_total = (last_y + 1) * (last_x + 1) - tier1_total
            for y_base in range(0, last_y + 1, search_stride):
                for x_base in range(0, last_x + 1, search_stride):
                    for dy in range(search_stride):
                        y = y_base + dy
                        if y > last_y:
                            continue
                        for dx in range(search_stride):
                            x = x_base + dx
                            if x > last_x or (
                                y % window_size == 0 and x % window_size == 0
                            ):
                                # Already covered by the block-aligned Tier 1 grid.
                                continue
                            report(2, tier2_total, y, x)
                            result = scan_window(y, x)
                            if result is not None:
                                return result
                            if timed_out():
                                return None

        return None

    @staticmethod
    def _payload_bit_len(num_chars: int) -> int:
        """Total packet bit length for a payload of `num_chars` characters (11 + N bytes)."""
        return (11 + num_chars) * 8
