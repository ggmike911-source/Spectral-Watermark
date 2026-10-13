import time

import numpy as np
from PIL import Image

from spectral_watermark.pipeline import WatermarkPipeline


def test_pipeline_crop_recovery():
    rng = np.random.default_rng(42)
    image = np.clip(rng.normal(128.0, 25.0, size=(256, 256)), 0.0, 255.0).astype(np.uint8)
    pipeline = WatermarkPipeline()
    payload = "TEST_PAYLOAD_2026"

    stego, bit_len = pipeline.embed_image(image, payload, alpha=0.45)

    crop_y, crop_x = 37, 45
    crop_h, crop_w = 160, 160
    crop = stego[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

    recovered = pipeline.extract_from_crop(crop, expected_bit_len=bit_len, search_stride=2)

    assert recovered == payload


def _make_unaligned_fixture():
    """Shared fixture: 256x256 synthetic texture with 'TOP_SECRET' embedded."""
    rng = np.random.default_rng(7)
    gy, gx = np.mgrid[0:256, 0:256]
    texture = (
        128.0
        + 40.0 * np.sin((gx + gy) / 16.0)
        + 25.0 * np.cos(gx / 9.0) * np.sin(gy / 13.0)
        + rng.normal(0.0, 18.0, size=(256, 256))
    )
    image = np.clip(texture, 0.0, 255.0).astype(np.uint8)
    pipeline = WatermarkPipeline()
    stego, bit_len = pipeline.embed_image(image, "TOP_SECRET", alpha=0.45)
    return pipeline, stego, bit_len


def test_unaligned_crop_recovery():
    pipeline, stego, bit_len = _make_unaligned_fixture()
    crop = stego[29 : 29 + 157, 41 : 41 + 163]
    recovered = pipeline.extract_from_crop(crop, expected_bit_len=bit_len, search_stride=2)
    assert recovered == "TOP_SECRET"


def test_arbitrary_dimension_crop_recovery():
    pipeline, stego, bit_len = _make_unaligned_fixture()
    crop = stego[31 : 31 + 173, 43 : 43 + 189]
    assert crop.shape == (173, 189)

    recovered = pipeline.extract_from_crop(crop, expected_bit_len=bit_len, search_stride=2)
    assert recovered == "TOP_SECRET"


def test_spectrum_caching_blind_scan(monkeypatch):
    pipeline, stego, _ = _make_unaligned_fixture()
    crop = stego[29 : 29 + 157, 41 : 41 + 163]
    original_rfft2 = np.fft.rfft2
    fft_calls = 0

    def counted_rfft2(*args, **kwargs):
        nonlocal fft_calls
        fft_calls += 1
        return original_rfft2(*args, **kwargs)

    monkeypatch.setattr(np.fft, "rfft2", counted_rfft2)
    progress = []
    start = time.perf_counter()
    recovered = pipeline.extract_from_crop(
        crop,
        expected_bit_len=None,
        search_stride=2,
        on_progress=lambda tier, checked, total, y, x: progress.append(checked),
    )
    elapsed = time.perf_counter() - start

    assert recovered == "TOP_SECRET"
    assert elapsed < 3.0
    # One additional full-canvas FFT is used by pilot scale estimation.
    assert fft_calls == progress[-1] + 1


def test_scaled_display_recovery():
    pipeline, stego, bit_len = _make_unaligned_fixture()
    scaled_img = np.asarray(
        Image.fromarray(stego).resize((320, 320), Image.Resampling.BICUBIC)
    )

    recovered = pipeline.extract_from_crop(
        scaled_img,
        expected_bit_len=bit_len,
        scales=[1.0, 1.25],
    )
    assert recovered == "TOP_SECRET"


def test_too_small_crop_rejection():
    pipeline = WatermarkPipeline()
    tiny = np.random.default_rng(123).integers(0, 255, size=(50, 50), dtype=np.uint8)
    assert pipeline.extract_from_crop(tiny, expected_bit_len=None) is None
    assert pipeline.extract_from_crop(tiny, expected_bit_len=96) is None
