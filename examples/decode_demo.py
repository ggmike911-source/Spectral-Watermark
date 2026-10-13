"""
decode_demo.py - Standalone Blind Watermark Decoder
Accepts any image (PNG, JPG, BMP - full or cropped), searches for embedded
watermarks with a two-tier hierarchical scan, and outputs the recovered
payload if found, else None.

Usage:
    python examples/decode_demo.py path/to/image.png
    python examples/decode_demo.py path/to/image.png --stride 2
    python examples/decode_demo.py path/to/image.png --bit-len 160
    python examples/decode_demo.py path/to/image.png --multi-scale
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from spectral_watermark import WatermarkPipeline

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class ScanProgressPrinter:
    """Throttled progress reporter: prints at most once per ~1 s and on tier changes."""

    def __init__(self, enabled: bool = True, min_interval: float = 1.0) -> None:
        self.enabled = enabled and sys.stdout.isatty()
        self.min_interval = min_interval
        self.start = time.perf_counter()
        self.last_print = 0.0
        self.last_tier = 0

    def __call__(self, tier: int, checked: int, total: int, y: int, x: int) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if tier == self.last_tier and now - self.last_print < self.min_interval:
            return
        self.last_print = now
        self.last_tier = tier
        elapsed = now - self.start
        rate = checked / elapsed if elapsed > 0 else 0.0
        eta = (total - checked) / rate if rate > 0 else float("inf")
        if eta == float("inf"):
            eta_str = "??"
        elif eta >= 3600:
            eta_str = f"{eta / 3600:.1f}h"
        else:
            eta_str = f"{eta:.0f}s"
        pct = 100.0 * checked / total if total else 0.0
        print(
            f"\r    Tier {tier}: {checked}/{total} windows ({pct:5.1f}%) "
            f"| {rate:6.0f} win/s | elapsed {elapsed:6.1f}s | ETA {eta_str}   ",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        if self.enabled:
            print()


def decode_image(
    image_path: str | Path,
    expected_bit_len: int | None = None,
    search_stride: int = 2,
    min_chars: int = 1,
    max_chars: int = 32,
    scales: list[float] | None = None,
    timeout: float | None = None,
    quiet: bool = False,
) -> tuple[str | None, float]:
    """
    Attempts to decode a watermark from an image.
    If expected_bit_len is not specified, the pipeline's dynamic length
    scanner evaluates payloads from min_chars to max_chars characters.
    Returns (recovered_payload_or_None, elapsed_seconds).
    """
    path = Path(image_path)
    if not path.is_file():
        print(f"[!] Error: Image file not found: {path}", file=sys.stderr)
        return None, 0.0

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[!] Error: Unsupported image format '{path.suffix}'. "
              f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        return None, 0.0

    try:
        img_array = np.array(Image.open(path).convert("RGB"))
    except (OSError, ValueError) as e:
        print(f"[!] Error loading image '{path}': {e}", file=sys.stderr)
        return None, 0.0

    pipeline = WatermarkPipeline(block_size=64)

    height, width = img_array.shape[:2]
    if not quiet:
        print("[*] Image Size     : "
              f"{width} x {height} px")
    if not quiet:
        scale_description = f", scales={scales}" if scales is not None else ""
        print("[*] Starting hierarchical scan "
              f"(Tier 1: block-aligned grid, Tier 2: strided sliding, "
              f"stride={search_stride}{scale_description})...")
    progress = ScanProgressPrinter(enabled=not quiet)
    start = time.perf_counter()
    payload = pipeline.extract_from_crop(
        img_array,
        expected_bit_len=expected_bit_len,
        search_stride=search_stride,
        min_chars=min_chars,
        max_chars=max_chars,
        scales=scales,
        on_progress=progress if not quiet else None,
        timeout=timeout,
        on_scale_detected=(
            lambda scale: print(
                f"[*] Detected Scale: {scale:.2f}x"
                f"{' (Auto-Normalizing...)' if abs(scale - 1.0) > 0.02 else ''}"
            )
            if not quiet
            else None
        ),
    )
    elapsed = time.perf_counter() - start
    progress.finish()
    return payload, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind Watermark Extractor: Recover embedded payloads from arbitrary images."
    )
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        default="stego.png",
        help="Path to the watermarked or cropped image (defaults to 'stego.png')",
    )
    parser.add_argument(
        "--bit-len",
        type=int,
        default=None,
        help="Explicit expected bitstream length if known (skips length scan).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Sliding-window search stride in pixels (default: 2).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=32,
        help="Maximum payload character length to scan when bit-len is omitted (default: 32).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Abort the scan after this many seconds and report no match (default: no limit).",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Also scan common display scales (1.25x, 1.5x, 0.8x, and 0.75x).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        action="append",
        default=[],
        help="Try an apparent source scale factor (repeatable, e.g. --scale 2.55).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress output.",
    )

    args = parser.parse_args()

    if not args.quiet:
        print("================================================================")
        print(" spectral-watermark: Standalone Decoder Utility")
        print("================================================================")
        print(f"[*] Target Image   : {args.image}")
        print(f"[*] Search Stride  : {args.stride} px")
        if args.timeout is not None:
            print(f"[*] Timeout        : {args.timeout:.1f} s")
        if args.multi_scale:
            print("[*] Multi-scale    : enabled")
        if args.scale:
            print(f"[*] Explicit Scales: {args.scale}")
        if args.bit_len:
            print(f"[*] Mode           : Direct Extraction ({args.bit_len} bits)")
        else:
            print(f"[*] Mode           : Blind Dynamic Length Scan (1 to {args.max_chars} chars)")

    recovered, elapsed = decode_image(
        image_path=args.image,
        expected_bit_len=args.bit_len,
        search_stride=args.stride,
        max_chars=args.max_chars,
        scales=(
            ([1.0, 1.25, 1.5, 0.8, 0.75] if args.multi_scale else []) + args.scale
            or None
        ),
        timeout=args.timeout,
        quiet=args.quiet,
    )

    if not args.quiet:
        print(f"[*] Elapsed Time   : {elapsed * 1000:.1f} ms")
        print("----------------------------------------------------------------")
    if recovered is not None:
        print(f"[SUCCESS] Recovered Payload: '{recovered}'")
    else:
        print("[RESULT] None (No valid watermark detected)")
    if not args.quiet:
        print("----------------------------------------------------------------")


if __name__ == "__main__":
    main()


# How to run:
# 1. Decode a specific image (e.g. your scenery stego image or any cropped file)
# .venv\Scripts\python examples\decode_demo.py path\to\my_photo.png

# 2. Faster scan if you know the exact bit length
# .venv\Scripts\python examples\decode_demo.py path\to\my_photo.png --bit-len 160

# 3. Fine-grained pixel-by-pixel search stride (stride=1)
# .venv\Scripts\python examples\decode_demo.py path\to\my_photo.png --stride 1
