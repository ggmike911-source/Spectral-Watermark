# spectral-watermark

Crop-resilient, blind image watermarking using 2D Fourier transforms and Reed–Solomon error correction. Embed a short UTF-8 payload into image luminance and recover it without the original cover image.

> **Interactive Notebook Guide:** [Run the offline RGB showcase and Python/CLI quickstart](https://github.com/ggmike911-source/Spectral-Watermark/blob/main/examples/quickstart_showcase.ipynb).

---

## Features

- **Crop resilience:** 64 × 64 spatial tiles repeat the encoded payload across the image.
- **Blind extraction:** Search an image or crop without access to the original cover.
- **Error correction:** Reed–Solomon encoding helps recover payloads after moderate distortion.
- **Color support:** Embed in RGB/RGBA images through floating-point ITU-R BT.601 luminance. Equal RGB-channel adjustments nominally preserve chroma; RGBA alpha is preserved. Grayscale images remain grayscale.
- **Distortion tools:** The CLI benchmark tests cropping, JPEG compression, blur, and noise.

---

## Installation

```bash
python -m pip install spectral-watermark
```

Requires Python 3.10 or later.

### Development install

```bash
git clone https://github.com/ggmike911-source/Spectral-Watermark.git
cd Spectral-Watermark
python -m pip install -e ".[dev]"
```

### Optional Flask demo

From a source checkout, install and run the optional web demo:

```bash
python -m pip install -e ".[web]"
python app.py
```

The demo listens on `http://127.0.0.1:5000`.

---

## Python API quickstart

The pipeline accepts NumPy arrays and Pillow images. The example below embeds a payload into an RGB cover, saves the color-preserving result, then extracts from an unaligned crop:

```python
import numpy as np
from PIL import Image

from spectral_watermark import WatermarkPipeline

with Image.open("cover.png") as image:
    cover = np.asarray(image.convert("RGB"))
pipeline = WatermarkPipeline()
stego, bit_length = pipeline.embed_image(cover, "SECRET_2026")
Image.fromarray(stego).save("stego.png")

# A crop with offsets not aligned to the 64-pixel tile grid.
crop = stego[19:219, 23:223]
payload = pipeline.extract_from_crop(
    crop,
    expected_bit_len=bit_length,
    search_stride=2,
)
print(payload)
```

`embed_image` returns `(stego_array, bit_length)`. Grayscale input returns a 2D `uint8` array; RGB and RGBA input return color arrays, with RGBA alpha unchanged. The blind extractor can omit `expected_bit_len` to scan candidate payload lengths.

---

## Command-line interface

The `spectral-watermark` command has three subcommands. All image commands accept `-i` / `--input`.

### Embed

```bash
spectral-watermark embed --input cover.png --payload "SECRET_2026" --output stego.png
```

- `-i`, `--input` (required): cover image.
- `-p`, `--payload` (required): non-empty UTF-8 payload that fits one tile.
- `-o`, `--output` (required): output image path. Parent directories are created.

The command reports image dimensions, encoded bit length, stego PSNR, and elapsed time. PNG or another lossless output format is recommended. JPEG output is allowed with a warning; RGBA saved to JPEG is converted to RGB and alpha is discarded.

### Extract

```bash
spectral-watermark extract --input cropped_stego.png
spectral-watermark extract --input screenshot.png --scale 1.25 --scale 1.5 --timeout 20
```

- `--stride` (default `2`): spatial sliding-window stride; must be at least `1`.
- `--bit-len` (optional): known encoded frame length in bits. If omitted, scan candidate lengths.
- `--min-chars` / `--max-chars` (defaults `1` / `32`): candidate character bounds for blind scanning.
- `--scale FACTOR` (repeatable): explicit apparent zoom factor to try.
- `--multi-scale`: also try preset factors `1.0`, `1.25`, `1.5`, `0.8`, and `0.75`.
- `--timeout SECONDS` (optional): maximum scan duration.
- `-q`, `--quiet`: print only the recovered payload to standard output.

### Benchmark

```bash
spectral-watermark benchmark --input cover.png
spectral-watermark benchmark --input cover.png --payload "TEST" --seed 123
```

- `-p`, `--payload` (default `SECRET_2026`): test payload.
- `--seed` (default `42`): deterministic seed for generated noise.

The benchmark reports baseline embedding PSNR and recovery results for a center crop, JPEG quality 75 and 50, Gaussian blur, and Gaussian noise. Individual attack failures are reported in the table; the command exits non-zero if baseline recovery fails.

---

## Technical constraints

- The CLI requires cover and extraction images to be at least **64 × 64 pixels**.
- The default tile size is **64 × 64 pixels**. Encoded data is repeated across tiles.
- Payload capacity is limited by the single-tile spectral-bin and Reed–Solomon frame capacity: approximately **29 UTF-8 bytes** (up to **320 encoded bits**) per tile. Multi-byte UTF-8 characters count by their encoded bytes, not their displayed character count.
- RGB/RGBA embedding adds the computed luminance change equally to red, green, and blue. Integer rounding and clipping at saturated color boundaries can cause small local chroma deviations.
- JPEG is lossy and may reduce recovery reliability. Use a lossless image format when possible.

---

## Development checks

```bash
pytest -v
ruff check src/ tests/ examples/
mypy src/
```

---

## License

MIT License. Copyright (c) 2026 Mike Xian (@ggmike911-source).
