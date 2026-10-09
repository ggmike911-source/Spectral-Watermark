# spectral-watermark

A high-performance, crop-resilient, blind image watermarking engine using 2D Discrete Fourier Transforms (2D FFT) and Reed-Solomon error correction.

## Project Architecture

1. **Spatial Block Tiling ($64 \times 64$)**: Redundant payload distribution across tiles to ensure crop resilience.
2. **Mid-Frequency Ring Modulation**: Bypasses low-frequency visual artifacts and high-frequency lossy compression quantization.
3. **Reed-Solomon Error Correction**: Mathematical correction of bit corruption and noise induced by post-processing.

## Installation & Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install with development dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```
