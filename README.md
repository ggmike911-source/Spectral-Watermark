# spectral-watermark

A high-performance, crop-resilient, blind image watermarking engine using 2D Discrete Fourier Transforms (2D FFT) and Reed-Solomon error correction.

---

## Key Features

1. **Spatial Block Tiling (64 × 64)**: Redundant payload distribution across independent blocks ensures complete payload recovery if at least one intact block survives cropping.
2. **Mid-Frequency Ring Modulation**: Injects secret bits into mid-frequency annular magnitude bands, bypassing low-frequency visual artifacts and surviving high-frequency JPEG lossy quantization.
3. **Reed-Solomon Error Correction**: Mathematical correction of corrupted or flipped bits induced by noise, blur, and lossy compression.
4. **Blind Extraction**: Recovers data directly from stego-images without requiring the original host image.

---

## Requirements

- Python 3.10+
- Operating System: Linux, macOS, or Windows

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/spectral-watermark.git
cd spectral-watermark

# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# 3. Upgrade pip and install package in editable mode with dev dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## Quick Usage

### Command-Line Interface (CLI)

```bash
# Embed a secret payload into an image
spectral-watermark embed --input cover.png --payload "SECRET_2026" --output stego.png

# Blindly extract payload from a watermarked or cropped image
spectral-watermark extract --input cropped_stego.png

# Run adversarial distortion benchmarks (Cropping, JPEG compression, Blur, Noise)
spectral-watermark benchmark --input cover.png
```

---

## Running Tests

Execute the automated test suite with coverage:

```bash
pytest
```

---

## License

This project is licensed under the MIT License.
