# Engineering Log

This log records project changes and verification outcomes. Commands use repository-relative paths and do not include local machine details.

---

## Phase 1 — Repository foundation

### Summary & goal
Set up the Python package layout and development workflow.

### Key technical changes
- Added the `src/spectral_watermark/` package, tests, examples, and project metadata.
- Configured setuptools, the `spectral-watermark` entry point, runtime dependencies, and ignore rules for generated files.

### Verification
- Verified imports for the package and its runtime dependencies.
- Confirmed the repository layout and development checks.

---

## Phase 2 — Core watermarking engine

### Summary & goal
Implement the first end-to-end blind watermarking flow.

### Key technical changes
- Added Reed–Solomon/UTF-8 packet framing, FFT-domain bit modulation, and tiled image embedding.
- Added initial tests for packet recovery, spectral-bin selection, image embedding, and crop recovery.
- Corrected the crop-search phase handling so non-block-aligned offsets are examined.

### Verification
- Initial test suite: **7 passed**.

---

## Phase 3 — Faster crop decoding

### Summary & goal
Reduce repeated FFT work and make blind extraction practical on unaligned crops.

### Key technical changes
- Added extraction from precomputed spectra and a sync-preamble fast rejection check.
- Added hierarchical aligned/strided scans and dynamic frame-length search.
- Updated `examples/decode_demo.py` to use the pipeline scanner rather than rescanning the image per candidate length.

### Verification
- Test suite at this stage: **10 passed**; Ruff passed.
- Recorded a **4.6×** speedup on the unaligned-crop blind-scan fixture compared with the earlier per-length scan.
- Verified recovery from a crop at offsets not aligned to the tile grid.

---

## Phase 4 — Scale-aware crop recovery

### Summary & goal
Improve extraction from resized or display-scaled images.

### Key technical changes
- Added optional scale scans with bicubic resampling and retained progress and timeout controls.
- Added regression coverage for arbitrary crop dimensions, spectrum caching, and scaled display recovery.

### Verification
- Test suite at this stage: **12 passed**; Ruff and whitespace checks passed.
- Confirmed a bounded multi-scale scan exits cleanly on a non-matching image.

---

## Phase 5 — Pilot calibration and border trimming

### Summary & goal
Estimate display scaling from embedded spectral pilots and remove solid screenshot borders.

### Key technical changes
- Added `PilotCalibrator`, symmetric pilot-bin injection, and safe solid-border trimming.
- Integrated calibration-first extraction with native-scale fallback and scale-reporting callbacks.
- Reserved pilot coordinates from payload-bin allocation.

### Verification
- Test suite at this stage: **15 passed**.
- The 1.25× fixture measured approximately **1.257×** at full size and **1.248×** on an unaligned crop.
- Calibration and crop-recovery regressions passed.

---

## Phase 6 — Screen-capture investigation

### Summary & goal
Investigate a screenshot that did not decode through automatic calibration.

### Key technical changes
- Changed scan ordering so calibrated and explicitly requested scales are tried before native-scale fallback.
- Added a repeatable explicit `--scale` option to the standalone decoder.
- Recorded the suspected scale only as a diagnostic lead; the bounded attempt was inconclusive and did not establish that the image was unwatermarked.

### Verification
- A known native-scale fixture recovered its payload in approximately **52 ms**.
- Test suite at this stage: **15 passed**; Ruff passed.
- Automatic arbitrary-scale recovery from that particular capture remains unconfirmed; current tests cover controlled scaling.

---

## Phase 7 — Click CLI and benchmark

### Summary & goal
Provide a supported command-line interface for embedding, extraction, and distortion benchmarks.

### Key technical changes
- Added the `embed`, `extract`, and `benchmark` commands with validation, helpful errors, timing, and quiet extraction output.
- Added in-memory crop, JPEG, blur, and noise benchmark cases.
- Added CLI tests and corrected the pipeline progress-callback signature.

### Verification
- Test suite at this stage: **23 passed**.
- Ruff and mypy passed; the installed `spectral-watermark --help` entry point listed all three commands.

---

## Phase 8 — RGB and RGBA support

### Summary & goal
Preserve image color while embedding into luminance, with grayscale behavior unchanged.

### Key technical changes
- Added floating-point BT.601 luminance embedding for RGB/RGBA and preserved RGBA alpha.
- Added support for palette, luminance-alpha, and CMYK Pillow inputs.
- Updated border trimming, CLI image handling, PSNR, and benchmark attacks for color images.

### Verification
- Test suite at this stage: **30 passed**; Ruff and mypy passed.
- A representative RGB fixture measured **36.08 dB** embedding PSNR and recovered after JPEG quality 75.
- Notebook RGB fixture measured **32.67 dB** PSNR with equal per-channel adjustments.

---

## Phase 9 (Part 1) — Packaging and PyPI documentation

### Summary & goal
Prepare the distribution metadata and package contents for publication.

### Key technical changes
- Added the PEP 561 `py.typed` marker and setuptools package-data configuration.
- Added an explicit manifest that includes release files and excludes development-only content.
- Updated SPDX license metadata, repository URLs, package description, Python requirement, and classifiers.
- Added the verified GitHub repository URL to the PyPI-ready README.

### Verification
- Built and inspected wheel and source-distribution contents; confirmed `py.typed`, license, README, and metadata were included and development files were excluded.
- Confirmed the project targets Python **3.10+** and advertises Python 3.10–3.12 classifiers.

---

## Phase 9 (Part 2) — Interactive notebook showcase

### Summary & goal
Provide an executable, offline showcase of the Python API and CLI.

### Key technical changes
- Added a notebook covering package imports, RGB embedding, distortion recovery, scale calibration, alpha preservation, and CLI usage.
- Sanitized the CLI example output so it shows a generic output filename rather than a local temporary path.

### Verification
- Re-executed all four code cells successfully.
- Confirmed recovery after an unaligned crop, JPEG quality 75, a 1.25× scaled screenshot with a border, and RGBA embedding.
- The scaled screenshot calibrated at approximately **1.24×**; its RGB fixture measured **32.67 dB** PSNR.

---

## Final Repository Audit & Polish

### Summary & goal
Complete the pre-submission audit, align public documentation with the implementation, and remove machine-specific details.

### Key technical changes
- Repaired missing commas in the pre-existing author-list edit in `pyproject.toml` without removing the additional author entries; removed the misleading cryptography classifier.
- Corrected blind extraction to scan UTF-8 byte lengths while enforcing the documented character bounds. Added a multibyte-payload regression test and invalid scan-argument checks.
- Removed an invalid shell command from `app.py`, added a dedicated Flask extra, aligned the web demo with the current scale API, and disabled debug mode.
- Made `examples/visual_demo.py` self-contained with a deterministic textured image; updated decoder argument validation, image-file cleanup, and example commands.
- Reformatted this log into consistent phase sections, removed local paths, environment-specific commands, and unrelated scratch notes, and standardized README section separators.
- Re-executed the notebook after replacing its printed temporary output path with a generic filename.

### Verification
- **Code and type safety — PASS:** all maintained Python files passed Ruff; `mypy src/` reported no issues in 6 source files; the full test suite passed **31 tests**.
- **CLI/API consistency — PASS:** inspected `--help` for the group and all three subcommands, confirmed README option names/defaults, and executed the README API example; it produced a `(256, 256, 3)` result, a 176-bit frame, and recovered `SECRET_2026`.
- **Capacity and metadata facts — PASS:** verified that 29 UTF-8 bytes fit at **320 bits** and 30 bytes exceed the single-tile capacity. README and project metadata agree on Python **3.10+** (classifiers 3.10–3.12), tile size, and color/alpha behavior.
- **Packaging and Git hygiene — PASS:** a clean build produced both wheel and sdist; `twine check --strict` returned **PASSED** for both. Archive inspection confirmed required metadata and `py.typed`, with tests, examples, logs, instructions, and image scratch files excluded. Generated build and distribution directories remain ignored.
- **Notebook sanitation — PASS:** all four code cells executed; saved cell outputs contain no local absolute paths or usernames.
- **Documentation and formatting — PASS:** README and log headings, section separators, and fenced-code language tags are consistent; recorded test, capacity, and RGB PSNR figures match verified runs.
- **Root demo smoke test — PASS:** `examples/visual_demo.py` recovered `TOP_SECRET` from its generated unaligned crop.
- Final validation: `pytest -v` (**31 passed in 24.45s**), `ruff check src/ tests/ examples/` (**passed**), `mypy src/` (**no issues**), installed CLI help (**passed**), clean `python -m build` (**wheel and sdist built**), and strict Twine checks (**both passed**).
