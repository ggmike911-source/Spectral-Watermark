import numpy as np

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

    recovered = pipeline.extract_from_crop(crop, bit_len, search_stride=2)

    assert recovered == payload
