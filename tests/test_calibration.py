import numpy as np
from PIL import Image

from spectral_watermark.calibration import PilotCalibrator, trim_solid_borders
from spectral_watermark.pipeline import WatermarkPipeline


def _make_stego():
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


def test_scale_estimation_accuracy():
    _, stego, _ = _make_stego()
    scaled = np.asarray(
        Image.fromarray(stego).resize((320, 320), Image.Resampling.BICUBIC)
    )

    estimated = PilotCalibrator().estimate_scale(scaled)
    assert abs(estimated - 1.25) <= 0.05


def test_scaled_and_unaligned_crop_recovery():
    pipeline, stego, _ = _make_stego()
    scaled = np.asarray(
        Image.fromarray(stego).resize((320, 320), Image.Resampling.BICUBIC)
    )
    crop = scaled[33 : 33 + 187, 47 : 47 + 205]

    assert pipeline.extract_from_crop(crop) == "TOP_SECRET"


def test_border_trimming():
    _, stego, _ = _make_stego()
    padded = np.pad(stego, ((30, 30), (30, 30)), constant_values=48)

    trimmed = trim_solid_borders(padded)
    assert trimmed.shape == stego.shape
    assert np.array_equal(trimmed, stego)
