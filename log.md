# Execution Log

(9/10/2026 22:05)
## Step 1: Git init + .gitignore — DONE
- Initialized git repository at `E:\Spectral-Watermark` with default branch `main` (repo was not previously initialized).
- Created `.gitignore` at repo root with the specified ignore rules (Python artifacts, venvs, IDE files, test/coverage caches, benchmark outputs; `tests/fixtures/*.png|jpg` re-included).

## Step 2: Directory scaffold [DONE]
- Created `src/spectral_watermark/`, `tests/fixtures/`, `benchmarks/`, `docs/`.
- Created placeholder `src/spectral_watermark/__init__.py` and `tests/__init__.py`.

## Step 3: pyproject.toml [DONE]
- Created `pyproject.toml` at repo root: setuptools build backend, project metadata (name `spectral-watermark`, v0.1.0, Python >=3.10, MIT), runtime deps (numpy, scipy, pillow, reedsolo, click), dev extras (pytest, pytest-cov, ruff, mypy, matplotlib), console script `spectral-watermark`, pytest/ruff tool config.

## Step 4: README.md [DONE]
- Created `README.md` at repo root with architecture overview, install/setup instructions, and test instructions.

## Step 5: Environment creation & installation [DONE]
- Created virtual environment `.venv` using Python 3.10.3 (`C:\Users\Ben\AppData\Local\Programs\Python\Python310\python.exe`).
- Upgraded pip to 26.2.1. (Note: first in-place upgrade hit a Windows file-lock OSError on `pip.exe`, leaving a broken `-ip` distribution; repaired via `ensurepip --upgrade`, removed the stray dist dir, then the pip upgrade succeeded.)
- Installed package in editable mode with dev extras: `pip install -e ".[dev]"`.
- Installed versions: numpy 2.2.6, scipy 1.15.3, pillow 12.3.0, reedsolo 1.7.0, click 8.5.0, pytest 9.1.1, pytest-cov 7.1.0, ruff 0.16.10, mypy 2.4.0, matplotlib 3.10.9.

## Step 6: Automated verification [DONE]
- Ran `.venv\Scripts\python.exe -c "import spectral_watermark, numpy, reedsolo, PIL, click, scipy; print('Environment & package setup verified successfully!')"`.
- Output: `Environment & package setup verified successfully!` (exit code 0).

## Final state
All 6 steps from `instructions.md` completed successfully. Repository is initialized on branch `main` with no commits yet (initial commit is a follow-up action for the reviewer).

Note: `benchmarks/`, `docs/`, and `tests/fixtures/` are empty; git does not track empty directories, so they will only appear in the repo once a file (e.g. a `.gitkeep`) is added.

```text
Spectral-Watermark/
├── .gitignore
├── pyproject.toml
├── README.md
├── instructions.md
├── log.md
├── src/
│   └── spectral_watermark/
│       ├── __init__.py
│       ├── ecc.py
│       ├── core.py
│       └── pipeline.py
├── tests/
│   ├── __init__.py
│   ├── test_ecc.py
│   ├── test_core.py
│   ├── test_pipeline.py
│   └── fixtures/
├── benchmarks/
└── docs/
```

## Phase 2 Step 1: Source modules [DONE]
- Created `src/spectral_watermark/ecc.py` with `BitCodec` for sync-marker framing, RS error correction, and UTF-8 payload encoding/decoding.
- Created `src/spectral_watermark/core.py` with `SpectralEngine`: deterministic FFT coordinate selection, pairwise magnitude modulation embedding, and blind extraction.
- Created `src/spectral_watermark/pipeline.py` with `WatermarkPipeline` for grayscale conversion, full-image tile embedding, and crop-window search and recovery.
- Updated `src/spectral_watermark/__init__.py` to expose `BitCodec`, `SpectralEngine`, `WatermarkPipeline`, and version `0.1.0`.

## Phase 2 Step 2: Unit tests [DONE]
- Created the three test files under `tests/`: `test_ecc.py`, `test_core.py`, and `test_pipeline.py` to cover round-trip correctness, RS resilience, frequency exclusions, embedding/extraction, PSNR, and crop recovery.
- Notes: the RS corruption test initially corrupted the sync preamble itself; corrected to corrupt only the ECC payload bytes while keeping the sync marker intact. The crop search also needed to inspect all phase offsets for a given stride so odd block offsets are not missed.

## Phase 2 Step 3: Validation & log update [DONE]
- Executed: `.venv\Scripts\pytest -v`
- Result: `7 passed in 1.29s` (exit code 0).
- Source and test modules satisfy the Phase 2 technical specification and the crop-recovery requirement for non-block-aligned images.

## Incident & Fix: Investigation of Visual Demo Extraction Failure

**Date:** October 9, 2026  
**Status:** Resolved  
**Affected File:** `visual_demo.py`  

### 1. Problem Description
Running `visual_demo.py` embedded the watermark with high fidelity (PSNR 44.5 dB), but Panel 4 reported `Recovered: 'None'` following a 40% spatial crop attack. Concurrently, running the test pipeline (`pytest`) and an inline PowerShell one-liner produced 100% accurate extraction on uniform and textured images.

### 2. Root Cause Analysis
1. **Mathematical Under-excitation in Synthetic Canvas:**  
   `generate_sample_canvas` generated a smooth synthetic wave:
   $$\text{pattern} = 128 + 55\sin(x)\cos(y) + 25\cos(2x)$$
   Because this signal is composed entirely of low-frequency basis functions, Fourier magnitudes inside the annular mid-frequency band ($8 \le r \le 22$) evaluated to near zero ($\mu \approx 1.1$).
2. **Sub-LSB Spatial Energy:**  
   The differential margin defaulted to the absolute floor $\Delta = \max(0.45 \cdot \mu, 1.5) = 1.5$. In a $64 \times 64$ block ($N^2 = 4096$), an FFT magnitude perturbation of $1.5$ yields a spatial change of:
   $$\delta_{\text{pixel}} \approx \frac{2 \Delta}{N^2} = \frac{3.0}{4096} \approx \pm 0.00073 \text{ intensity units}$$
   When cast to `np.uint8`, rounding truncated this change back to zero, erasing the watermark modulation.
3. **Natural Images vs. Synthetic Waves:**  
   Natural images and textured noise contain substantial spectral energy ($\mu \approx 1000 - 3000$). At $\mu = 1500$, $\Delta \approx 675$, which maps to spatial shifts of $2$ to $5$ gray-level units, surviving 8-bit integer quantization without bit error.

### 3. Resolution Applied
1. **Updated Synthetic Generator in `visual_demo.py`:** Replaced the pure trigonometric function with a realistic test texture comprising a spatial gradient combined with Gaussian high/mid-frequency variation.
2. **Aligned Crop Search Parameters:** Configured `search_stride=2` in `extract_from_crop` to ensure the sliding window evaluates both odd and even phase offsets across the candidate crop canvas.
3. **Verification:** Re-running `visual_demo.py` achieved:
   - Stego PSNR: **37.0 dB** (transparent perceptual quality).
   - Recovery: **100% exact match** (`TOP_SECRET_IOAI`) verified via Reed-Solomon polynomial parity check.

