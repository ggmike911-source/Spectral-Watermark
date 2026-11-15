from __future__ import annotations

import io
import math
import time
from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

from . import __version__
from .pipeline import WatermarkPipeline

_MIN_IMAGE_SIZE = 64
_MULTI_SCALES = [1.0, 1.25, 1.5, 0.8, 0.75]


def _load_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode in {"L", "RGB", "RGBA"}:
                return np.asarray(image).copy()
            if image.mode in {"LA", "PA"} or "transparency" in image.info:
                return np.asarray(image.convert("RGBA")).copy()
            if image.mode in {"CMYK", "YCbCr"}:
                return np.asarray(image.convert("RGB")).copy()
            if image.mode == "P":
                return np.asarray(image.convert("RGB")).copy()
            return np.asarray(image.convert("L")).copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise click.ClickException(f"Cannot read image '{path}': {exc}") from exc


def _validate_image_size(image: np.ndarray, path: Path, minimum: int = _MIN_IMAGE_SIZE) -> None:
    height, width = image.shape[:2]
    if height < minimum or width < minimum:
        raise click.ClickException(
            f"Image '{path}' is too small ({width}x{height}); minimum size is "
            f"{minimum}x{minimum} pixels."
        )


def _compute_psnr(reference: np.ndarray, degraded: np.ndarray) -> float:
    if reference.ndim == 3 and reference.shape[2] == 4:
        reference = reference[..., :3]
        degraded = degraded[..., :3]
    difference = reference.astype(np.float64) - degraded.astype(np.float64)
    mse = float(np.mean(difference**2))
    return math.inf if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def _validate_payload(pipeline: WatermarkPipeline, payload: str) -> int:
    if not payload:
        raise click.ClickException("Payload must not be empty.")
    try:
        bits = pipeline.codec.encode_payload(payload)
        pipeline.engine.get_bin_pairs(len(bits))
    except (TypeError, ValueError) as exc:
        raise click.ClickException(f"Payload cannot fit in a single tile: {exc}") from exc
    return len(bits)


@click.group()
@click.version_option(version=__version__, prog_name="spectral-watermark")
def cli() -> None:
    """Embed and extract crop-resilient spectral watermarks."""


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input cover image.",
)
@click.option("-p", "--payload", required=True, type=str, help="UTF-8 secret string to embed.")
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination for the watermarked image.",
)
def embed(input_path: Path, payload: str, output_path: Path) -> None:
    """Embed a payload into an image."""
    original = _load_image(input_path)
    _validate_image_size(original, input_path)

    pipeline = WatermarkPipeline()
    bit_len = _validate_payload(pipeline, payload)

    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        click.echo(
            "Warning: JPEG output is lossy and may reduce watermark recovery margin.",
            err=True,
        )

    started = time.perf_counter()
    try:
        stego, bit_len = pipeline.embed_image(original, payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image = Image.fromarray(stego)
        if output_image.mode == "RGBA" and output_path.suffix.lower() in {".jpg", ".jpeg"}:
            click.echo(
                "Note: JPEG does not support alpha; saving the RGB channels only.",
                err=True,
            )
            output_image = output_image.convert("RGB")
        output_image.save(output_path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Could not embed or save watermark: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    psnr = _compute_psnr(original, stego)

    click.echo(f"Watermarked image saved to: {output_path}")
    click.echo(f"Dimensions: {stego.shape[1]}x{stego.shape[0]}")
    click.echo(f"Payload bit length: {bit_len}")
    click.echo(f"Stego PSNR: {psnr:.2f} dB")
    click.echo(f"Elapsed: {elapsed_ms:.2f} ms")


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Watermarked or cropped image.",
)
@click.option("--stride", default=2, show_default=True, type=click.IntRange(min=1))
@click.option("--bit-len", default=None, type=click.IntRange(min=1), help="Known encoded frame bit length.")
@click.option("--min-chars", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--max-chars", default=32, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--scale",
    "scales",
    multiple=True,
    type=float,
    help="Explicit apparent zoom factor to try; may be repeated.",
)
@click.option("--multi-scale", is_flag=True, default=False, help="Try the preset scale sweep.")
@click.option("--timeout", default=None, type=click.FloatRange(min=0.0, min_open=True))
@click.option("-q", "--quiet", is_flag=True, default=False, help="Print only the recovered payload.")
def extract(
    input_path: Path,
    stride: int,
    bit_len: int | None,
    min_chars: int,
    max_chars: int,
    scales: tuple[float, ...],
    multi_scale: bool,
    timeout: float | None,
    quiet: bool,
) -> None:
    """Extract a payload from a watermarked image or crop."""
    image = _load_image(input_path)
    _validate_image_size(image, input_path)
    if min_chars > max_chars:
        raise click.ClickException("--min-chars must be less than or equal to --max-chars.")
    if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
        raise click.ClickException("Every --scale value must be finite and greater than zero.")

    scan_scales = list(scales)
    if multi_scale:
        scan_scales.extend(_MULTI_SCALES)

    def report_scale(scale: float) -> None:
        if not quiet:
            click.echo(f"Detected apparent scale: {scale:.3f}x")

    pipeline = WatermarkPipeline()
    started = time.perf_counter()
    try:
        recovered = pipeline.extract_from_crop(
            image,
            expected_bit_len=bit_len,
            search_stride=stride,
            scales=scan_scales,
            min_chars=min_chars,
            max_chars=max_chars,
            timeout=timeout,
            on_scale_detected=report_scale,
        )
    except ValueError as exc:
        raise click.ClickException(f"Could not extract watermark: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if recovered is None:
        message = "Payload not found"
        if timeout is not None and elapsed_ms >= timeout * 1000.0:
            message += f" before the {timeout:g}s timeout"
        else:
            message += " in the image"
        raise click.ClickException(message + ".")

    if quiet:
        click.echo(recovered)
        return
    click.echo(f"Recovered payload: {recovered}")
    click.echo(f"Elapsed: {elapsed_ms:.2f} ms")


def _jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    jpeg_input = image[..., :3] if image.ndim == 3 and image.shape[2] == 4 else image
    Image.fromarray(jpeg_input).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        compressed.load()
        if image.ndim == 2:
            result = compressed.convert("L")
        else:
            result = compressed.convert("RGB")
        return np.asarray(result).copy()


def _format_table(rows: list[tuple[str, str, str, str, str]]) -> str:
    headers = ("Attack Name", "Attack PSNR / Param", "Recovered", "Exact Match", "Time (ms)")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join((header, divider, *body))


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Cover image to benchmark.",
)
@click.option("-p", "--payload", default="SECRET_2026", show_default=True, type=str)
@click.option("--seed", default=42, show_default=True, type=int)
def benchmark(input_path: Path, payload: str, seed: int) -> None:
    """Benchmark recovery under deterministic image distortions."""
    original = _load_image(input_path)
    _validate_image_size(original, input_path)

    pipeline = WatermarkPipeline()
    bit_len = _validate_payload(pipeline, payload)
    try:
        stego, bit_len = pipeline.embed_image(original, payload)
    except ValueError as exc:
        raise click.ClickException(f"Could not embed benchmark payload: {exc}") from exc

    height, width = stego.shape[:2]
    crop_height = max(64, round(height / math.sqrt(2.0)))
    crop_width = max(64, round(width / math.sqrt(2.0)))
    crop_y = max((height - crop_height) // 2, 0)
    crop_x = max((width - crop_width) // 2, 0)
    if height - crop_height > 0 and crop_y % 2 == 0:
        crop_y += 1
    if width - crop_width > 0 and crop_x % 2 == 0:
        crop_x += 1
    crop = stego[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]

    attack_image = stego[..., :3] if stego.ndim == 3 and stego.shape[2] == 4 else stego
    blurred = np.asarray(
        Image.fromarray(attack_image).filter(ImageFilter.GaussianBlur(radius=0.8))
    )
    rng = np.random.default_rng(seed)
    noisy = np.clip(
        attack_image.astype(np.float64)
        + rng.normal(0.0, 3.0, size=attack_image.shape),
        0.0,
        255.0,
    ).astype(np.uint8)

    attacks: list[tuple[str, np.ndarray, str]] = [
        ("Baseline (No Attack)", stego, f"{_compute_psnr(original, stego):.2f} dB"),
        ("Center / Unaligned Crop", crop, "50% area"),
        ("JPEG Compression", _jpeg_round_trip(stego, 75), "quality=75"),
        ("JPEG Compression", _jpeg_round_trip(stego, 50), "quality=50"),
        ("Gaussian Blur", blurred, "radius=0.8"),
        ("Gaussian Noise", noisy, "sigma=3.0"),
    ]
    table_rows: list[tuple[str, str, str, str, str]] = []
    baseline_passed = False

    for index, (name, attacked, parameter) in enumerate(attacks):
        started = time.perf_counter()
        try:
            recovered = pipeline.extract_from_crop(
                attacked,
                expected_bit_len=bit_len,
                search_stride=2,
            )
        except ValueError as exc:
            raise click.ClickException(f"Benchmark extraction failed for {name}: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        exact_match = recovered == payload
        if index == 0:
            baseline_passed = exact_match

        attack_metric = parameter
        if index > 1:
            attack_metric = f"{parameter}; {_compute_psnr(stego, attacked):.2f} dB"

        table_rows.append(
            (
                name,
                attack_metric,
                "Yes" if recovered is not None else "No",
                "PASS" if exact_match else "FAIL",
                f"{elapsed_ms:.2f}",
            )
        )

    click.echo(f"Baseline embedding PSNR: {_compute_psnr(original, stego):.2f} dB")
    click.echo(_format_table(table_rows))
    if not baseline_passed:
        raise click.ClickException("Baseline payload recovery failed.")


main = cli


if __name__ == "__main__":
    cli()
