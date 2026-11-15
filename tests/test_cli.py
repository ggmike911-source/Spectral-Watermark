from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from spectral_watermark.cli import cli


@pytest.fixture
def textured_image(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    gy, gx = np.mgrid[0:256, 0:256]
    texture = (
        128.0
        + 40.0 * np.sin((gx + gy) / 16.0)
        + 25.0 * np.cos(gx / 9.0) * np.sin(gy / 13.0)
        + rng.normal(0.0, 18.0, size=(256, 256))
    )
    image_path = tmp_path / "cover.png"
    Image.fromarray(np.clip(texture, 0.0, 255.0).astype(np.uint8)).save(image_path)
    return image_path


def test_cli_version_and_help() -> None:
    runner = CliRunner()

    version = runner.invoke(cli, ["--version"])
    help_result = runner.invoke(cli, ["--help"])

    assert version.exit_code == 0
    assert "spectral-watermark, version" in version.output
    assert help_result.exit_code == 0
    assert "embed" in help_result.output
    assert "extract" in help_result.output
    assert "benchmark" in help_result.output


def test_cli_embed_and_extract_roundtrip(textured_image: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    stego_path = tmp_path / "nested" / "stego.png"

    embedded = runner.invoke(
        cli,
        [
            "embed",
            "--input",
            str(textured_image),
            "--payload",
            "SECRET_2026",
            "--output",
            str(stego_path),
        ],
    )
    extracted = runner.invoke(cli, ["extract", "--input", str(stego_path), "--quiet"])

    assert embedded.exit_code == 0, embedded.output
    assert "Stego PSNR:" in embedded.output
    assert stego_path.exists()
    assert extracted.exit_code == 0, extracted.output
    assert extracted.output.strip() == "SECRET_2026"


def test_cli_extract_unaligned_crop(textured_image: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    stego_path = tmp_path / "stego.png"
    crop_path = tmp_path / "crop.png"
    embedded = runner.invoke(
        cli,
        [
            "embed",
            "--input",
            str(textured_image),
            "--payload",
            "TOP_SECRET",
            "--output",
            str(stego_path),
        ],
    )
    assert embedded.exit_code == 0, embedded.output

    with Image.open(stego_path) as stego:
        stego.crop((23, 19, 23 + 156, 19 + 150)).save(crop_path)
    extracted = runner.invoke(cli, ["extract", "--input", str(crop_path), "--timeout", "10"])

    assert extracted.exit_code == 0, extracted.output
    assert "TOP_SECRET" in extracted.output


def test_cli_extract_unwatermarked_returns_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    tiny_path = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((50, 50), dtype=np.uint8)).save(tiny_path)

    result = runner.invoke(cli, ["extract", "--input", str(tiny_path)])

    assert result.exit_code != 0
    assert "too small" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("payload", ["", "x" * 30])
def test_cli_rejects_empty_or_over_capacity_payload(
    textured_image: Path, tmp_path: Path, payload: str
) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "embed",
            "--input",
            str(textured_image),
            "--payload",
            payload,
            "--output",
            str(tmp_path / "stego.png"),
        ],
    )

    assert result.exit_code != 0
    assert "Payload" in result.output
    assert "Traceback" not in result.output


def test_cli_invalid_image_format_returns_clean_error(tmp_path: Path) -> None:
    runner = CliRunner()
    invalid_path = tmp_path / "not-an-image.png"
    invalid_path.write_text("not an image", encoding="utf-8")

    result = runner.invoke(cli, ["extract", "--input", str(invalid_path)])

    assert result.exit_code != 0
    assert "Cannot read image" in result.output
    assert "Traceback" not in result.output


def test_cli_benchmark_runs(textured_image: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["benchmark", "--input", str(textured_image)])

    assert result.exit_code == 0, result.output
    assert "Attack Name" in result.output
    assert "Attack PSNR / Param" in result.output
    assert "Exact Match" in result.output
    assert "Baseline (No Attack)" in result.output
    assert "Center / Unaligned Crop" in result.output
