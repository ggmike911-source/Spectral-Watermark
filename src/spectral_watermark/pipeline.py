from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

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

    def embed_image(self, image: np.ndarray, payload: str, alpha: float = 0.45) -> Tuple[np.ndarray, int]:
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
        expected_bit_len: int,
        search_stride: int = 2,
    ) -> Optional[str]:
        """Search sliding windows across a crop and return the first valid decoded payload."""
        if search_stride <= 0:
            raise ValueError("search_stride must be positive")

        image = self._to_grayscale(cropped_image)
        block_size = self.engine.block_size
        pairs_a, pairs_b = self.engine.get_bin_pairs(expected_bit_len)

        height, width = image.shape
        for y_offset in range(search_stride):
            for y in range(y_offset, height - block_size + 1, search_stride):
                for x_offset in range(search_stride):
                    for x in range(x_offset, width - block_size + 1, search_stride):
                        window = image[y : y + block_size, x : x + block_size]
                        bits = self.engine.extract_bits(window, pairs_a, pairs_b)
                        decoded = self.codec.decode_packet(bits)
                        if decoded is not None:
                            return decoded

        return None
