import numpy as np

from spectral_watermark.ecc import BitCodec


def test_encode_decode_round_trip():
    codec = BitCodec()
    payload = "HELLO_SPECTRAL_WATERMARK"

    bits = codec.encode_payload(payload)

    assert bits.dtype == np.uint8
    assert set(np.unique(bits)).issubset({0, 1})
    assert codec.decode_packet(bits) == payload


def test_rs_correction_resilience():
    codec = BitCodec()
    payload = "A longer payload with accents: café 你好"

    bits = codec.encode_payload(payload)
    packet_bytes = np.packbits(bits, bitorder="big").tobytes()
    corrupted = bytearray(packet_bytes)
    start = len(codec.sync_marker)

    for offset in range(start, min(start + 3, len(corrupted))):
        corrupted[offset] ^= 0xFF

    corrupted_bits = np.unpackbits(
        np.frombuffer(corrupted, dtype=np.uint8),
        bitorder="big",
    ).astype(np.uint8)

    assert codec.decode_packet(corrupted_bits) == payload


def test_sync_marker_rejection():
    codec = BitCodec()
    bits = codec.encode_payload("payload")
    corrupted = bits.copy()
    corrupted[:16] = 1 - corrupted[:16]

    assert codec.decode_packet(corrupted) is None
