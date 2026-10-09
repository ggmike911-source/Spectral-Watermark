import numpy as np

from spectral_watermark.core import SpectralEngine


def test_get_bin_pairs_determinism_and_frequency_exclusion():
    engine = SpectralEngine(block_size=64, seed=2026)
    pairs_a, pairs_b = engine.get_bin_pairs(16)

    assert len(pairs_a) == 16
    assert len(pairs_b) == 16
    assert len(set(pairs_a).union(pairs_b)) == 32

    for pair in pairs_a + pairs_b:
        u, v = pair
        assert v != 0
        assert v != 32

    engine_repeat = SpectralEngine(block_size=64, seed=2026)
    pairs_a_repeat, pairs_b_repeat = engine_repeat.get_bin_pairs(16)
    assert pairs_a == pairs_a_repeat
    assert pairs_b == pairs_b_repeat


def test_single_tile_embedding_and_extraction():
    engine = SpectralEngine(block_size=64, seed=2026)
    tile = np.full((64, 64), 128.0, dtype=np.float64)
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)

    pairs_a, pairs_b = engine.get_bin_pairs(len(bits))
    embedded = engine.embed_tile(tile, bits, pairs_a, pairs_b, alpha=0.45)
    extracted = engine.extract_bits(embedded, pairs_a, pairs_b)

    assert np.array_equal(extracted, bits)


def test_psnr_preservation():
    rng = np.random.default_rng(0)
    tile = np.clip(rng.normal(128.0, 22.0, size=(64, 64)), 0.0, 255.0).astype(np.float64)
    engine = SpectralEngine(block_size=64, seed=2026)
    bits = np.array(
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
        dtype=np.uint8,
    )

    pairs_a, pairs_b = engine.get_bin_pairs(len(bits))
    embedded = engine.embed_tile(tile, bits, pairs_a, pairs_b, alpha=0.35)

    mse = np.mean((tile - embedded) ** 2)
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))

    assert psnr >= 38.0
