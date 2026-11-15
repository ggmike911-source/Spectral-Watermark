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


def _make_rgb_texture(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(23)
    gy, gx = np.mgrid[0:size, 0:size]
    texture = np.stack(
        (
            128.0 + 46.0 * np.sin(gx / 10.0) + 32.0 * np.cos(gy / 14.0),
            126.0 + 45.0 * np.sin((gx + gy) / 16.0) + 28.0 * np.cos(gx / 12.0),
            124.0 + 42.0 * np.cos(gx / 11.0) * np.sin(gy / 15.0) + 30.0 * np.sin(gy / 9.0),
        ),
        axis=2,
    )
    texture += rng.normal(0.0, 9.0, size=(size, size, 3))
    return np.clip(texture, 8.0, 247.0).astype(np.uint8)


def _ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color = rgb.astype(np.float64)
    red, green, blue = color[..., 0], color[..., 1], color[..., 2]
    return (
        0.299 * red + 0.587 * green + 0.114 * blue,
        -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0,
        0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0,
    )


def _rgb_psnr(original: np.ndarray, stego: np.ndarray) -> float:
    difference = original.astype(np.float64) - stego.astype(np.float64)
    mse = float(np.mean(difference**2))
    return float("inf") if mse == 0.0 else 20.0 * np.log10(255.0 / np.sqrt(mse))


def test_rgb_embed_preserves_color_and_recovers():
    original = _make_rgb_texture()
    pipeline = WatermarkPipeline()
    stego, _ = pipeline.embed_image(original, "RGB_SECRET")

    _, original_cb, original_cr = _ycbcr(original)
    _, stego_cb, stego_cr = _ycbcr(stego)

    assert stego.shape == (256, 256, 3)
    assert _rgb_psnr(original, stego) > 32.0
    assert float(np.mean(np.abs(original_cb - stego_cb))) < 0.5
    assert float(np.mean(np.abs(original_cr - stego_cr))) < 0.5
    assert pipeline.extract_from_crop(stego) == "RGB_SECRET"


def test_rgb_unaligned_crop_and_jpeg_recovery():
    from io import BytesIO

    original = _make_rgb_texture()
    pipeline = WatermarkPipeline()
    stego, bit_len = pipeline.embed_image(original, "RGB_SECRET")
    crop = stego[23 : 23 + 165, 37 : 37 + 171]

    encoded = BytesIO()
    Image.fromarray(stego).save(encoded, format="JPEG", quality=75)
    encoded.seek(0)
    with Image.open(encoded) as jpeg:
        jpeg.load()
        jpeg_rgb = np.asarray(jpeg.convert("RGB"))

    assert pipeline.extract_from_crop(
        crop, expected_bit_len=bit_len, search_stride=2
    ) == "RGB_SECRET"
    assert pipeline.extract_from_crop(
        jpeg_rgb, expected_bit_len=bit_len, search_stride=2
    ) == "RGB_SECRET"


def test_rgba_preserves_alpha_channel():
    rgb = _make_rgb_texture(128)
    gy, gx = np.mgrid[0:128, 0:128]
    alpha = ((gx + 2 * gy) % 256).astype(np.uint8)
    original = np.dstack((rgb, alpha))
    pipeline = WatermarkPipeline()

    stego, _ = pipeline.embed_image(original, "ALPHA_SECRET")

    assert stego.shape == (128, 128, 4)
    assert np.array_equal(stego[:, :, 3], original[:, :, 3])
    assert pipeline.extract_from_crop(stego) == "ALPHA_SECRET"


def test_rgb_saturated_colors_and_pil_modes():
    from PIL import Image

    rng = np.random.default_rng(42)
    gy, gx = np.mgrid[0:256, 0:256]
    texture = (
        18.0 * np.sin((gx + gy) / 9.0)
        + 12.0 * np.cos(gx / 7.0) * np.sin(gy / 11.0)
        + rng.normal(0.0, 4.0, size=(256, 256))
    )
    saturated = np.stack(
        (
            np.clip(250.0 + texture, 0.0, 255.0),
            np.clip(20.0 + texture, 0.0, 60.0),
            np.clip(10.0 + texture, 0.0, 50.0),
        ),
        axis=2,
    ).astype(np.uint8)
    pipeline = WatermarkPipeline()
    saturated_stego, _ = pipeline.embed_image(saturated, "RGB_SECRET")
    assert pipeline.extract_from_crop(saturated_stego) == "RGB_SECRET"

    palette = [
        component for value in range(256) for component in (value, 255 - value, value // 2)
    ]
    paletted = Image.fromarray(
        rng.integers(0, 256, (256, 256), dtype=np.uint8)
    ).convert("P")
    paletted.putpalette(palette)
    palette_stego, _ = pipeline.embed_image(paletted, "PALETTE_SECRET")
    assert palette_stego.shape == (256, 256, 3)
    assert pipeline.extract_from_crop(palette_stego) == "PALETTE_SECRET"

    alpha_gradient = np.arange(256, dtype=np.uint8)[None, :].repeat(256, axis=0)
    luminance_alpha = Image.fromarray(np.dstack((saturated[:, :, 0], alpha_gradient)))
    la_stego, _ = pipeline.embed_image(luminance_alpha, "LA_SECRET")
    assert la_stego.shape == (256, 256, 4)
    assert np.array_equal(la_stego[:, :, 3], np.asarray(luminance_alpha)[:, :, 1])
    assert pipeline.extract_from_crop(la_stego) == "LA_SECRET"


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
