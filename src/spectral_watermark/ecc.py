from __future__ import annotations

import numpy as np
from reedsolo import ReedSolomonError, RSCodec


class BitCodec:
    """Encode and decode a UTF-8 payload with a sync marker and Reed-Solomon ECC."""

    SYNC_MARKER = bytes([0x5A, 0xA5])

    def __init__(self, parity_bytes: int = 8, sync_marker: bytes | None = None) -> None:
        if parity_bytes <= 0:
            raise ValueError("parity_bytes must be positive")
        self.parity_bytes = int(parity_bytes)
        self.sync_marker = bytes(sync_marker) if sync_marker is not None else self.SYNC_MARKER
        self.codec = RSCodec(self.parity_bytes)

    def encode_payload(self, text: str) -> np.ndarray:
        """Encode a UTF-8 payload into a 1D bitstream with a sync marker and ECC."""
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        payload = text.encode("utf-8")
        if len(payload) > 255:
            raise ValueError("Payload length exceeds 255 bytes")

        frame = bytes([len(payload)]) + payload
        ecc_data = bytes(self.codec.encode(frame))
        packet = self.sync_marker + ecc_data
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8), bitorder="big").astype(np.uint8)
        return bits

    def decode_packet(self, bitstream: np.ndarray) -> str | None:
        """Decode a bitstream packet. Returns the original string on success, otherwise None."""
        flat = np.asarray(bitstream, dtype=np.uint8).reshape(-1)
        if flat.size < len(self.sync_marker) * 8:
            return None

        try:
            packet = np.packbits(flat, bitorder="big").tobytes()
        except ValueError:
            return None

        if len(packet) < len(self.sync_marker):
            return None
        if packet[: len(self.sync_marker)] != self.sync_marker:
            return None

        encoded_payload = packet[len(self.sync_marker) :]
        try:
            decoded = self.codec.decode(encoded_payload)
        except (ReedSolomonError, ValueError, IndexError):
            return None

        if isinstance(decoded, tuple):
            decoded = decoded[0]

        if not isinstance(decoded, (bytes, bytearray)):
            return None

        decoded = bytes(decoded)
        if len(decoded) < 1:
            return None

        payload_length = decoded[0]
        if payload_length > len(decoded) - 1:
            return None

        payload = decoded[1 : 1 + payload_length]
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return None
