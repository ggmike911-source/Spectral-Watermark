pip install flask

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image

from spectral_watermark import WatermarkPipeline

app = Flask(__name__)
pipeline = WatermarkPipeline()

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spectral Watermark Demo</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 900px;
      margin: 32px auto;
      padding: 0 16px;
      background: #f6f7f9;
      color: #1f2937;
    }
    h1 { margin-bottom: 6px; }
    .subtitle { color: #4b5563; margin-bottom: 24px; }
    .note {
      background: #fffbeb;
      border-left: 4px solid #f59e0b;
      padding: 12px 14px;
      border-radius: 6px;
      margin-bottom: 24px;
      font-size: 14px;
    }
    .card {
      background: #ffffff;
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    label { display: block; font-weight: 600; margin-top: 12px; margin-bottom: 4px; }
    input[type="text"], input[type="number"], input[type="file"] {
      width: 100%;
      box-sizing: border-box;
      padding: 8px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      background: #fff;
    }
    .inline-check { margin-top: 12px; display: flex; align-items: center; gap: 8px; }
    .inline-check input { width: auto; }
    button {
      margin-top: 16px;
      background: #2563eb;
      color: white;
      border: none;
      padding: 10px 18px;
      border-radius: 6px;
      font-size: 15px;
      cursor: pointer;
    }
    button:hover { background: #1d4ed8; }
    .result-box {
      margin-top: 16px;
      padding: 14px;
      border-radius: 8px;
      background: #f3f4f6;
    }
    .error {
      background: #fef2f2;
      color: #b91c1c;
      border-left: 4px solid #ef4444;
    }
    img.preview {
      max-width: 100%;
      max-height: 320px;
      border-radius: 6px;
      margin-top: 10px;
      border: 1px solid #e5e7eb;
    }
    code { background: #e5e7eb; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Spectral Watermark Interactive Demo</h1>
  <p class="subtitle">Embed a short payload into an image and test blind recovery after cropping</p>

  <div class="note">
    <strong>Notes:</strong>
    Input images must be at least <code>64x64</code> pixels. Use <strong>PNG</strong> (lossless)
    for best reliability, and keep payloads within roughly <strong>29 UTF-8 bytes</strong> per tile.
  </div>

  <!-- 1. Embed Watermark -->
  <div class="card">
    <h2>1. Embed Watermark</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="hidden" name="action" value="embed">

      <label>Cover Image (PNG / JPG / BMP, at least 64x64)</label>
      <input type="file" name="cover_image" accept="image/*" required>

      <label>Secret Payload</label>
      <input type="text" name="payload" placeholder="e.g. SECRET_2026" required>

      <button type="submit">Embed Watermark</button>
    </form>

    {% if embed_error %}
      <div class="result-box error">{{ embed_error }}</div>
    {% endif %}

    {% if stego_data_url %}
      <div class="result-box">
        <div><strong>Embedded successfully.</strong> Encoded <code>bit_length</code>: <code>{{ bit_length }}</code></div>
        <img class="preview" src="{{ stego_data_url }}" alt="Stego Image">
        <div style="margin-top: 10px;">
          <a href="{{ stego_data_url }}" download="stego_output.png">Download Stego Image (stego_output.png)</a>
        </div>
      </div>
    {% endif %}
  </div>

  <!-- 2. Quick Auto-Crop Test -->
  <div class="card">
    <h2>2. Quick Auto-Crop Test</h2>
    <p style="color:#4b5563; margin-top:0;">Upload a watermarked image to automatically crop a center region and test blind extraction.</p>
    <form method="post" enctype="multipart/form-data">
      <input type="hidden" name="action" value="auto_crop_extract">

      <label>Watermarked Image</label>
      <input type="file" name="crop_source_image" accept="image/*" required>

      <label>Center Crop Ratio (e.g. 0.65 keeps 65% of width and height)</label>
      <input type="number" name="crop_ratio" step="0.05" min="0.2" max="1.0" value="0.65">

      <label>Optional Known bit_length (leave blank for blind scan)</label>
      <input type="number" name="crop_bit_len" placeholder="e.g. 160 (optional)">

      <button type="submit">Crop and Extract</button>
    </form>

    {% if autocrop_error %}
      <div class="result-box error">{{ autocrop_error }}</div>
    {% endif %}

    {% if autocrop_result is not none %}
      <div class="result-box">
        <div><strong>Cropped Preview:</strong></div>
        <img class="preview" src="{{ autocrop_data_url }}" alt="Auto Cropped Image">
        <div style="margin-top: 10px;">
          <strong>Recovered Payload:</strong> <code>{{ autocrop_result }}</code>
        </div>
      </div>
    {% endif %}
  </div>

  <!-- 3. Blind Extraction from Uploaded Image -->
  <div class="card">
    <h2>3. Blind Extraction (Extract)</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="hidden" name="action" value="extract">

      <label>Query Image (full image, crop, or screenshot)</label>
      <input type="file" name="query_image" accept="image/*" required>

      <label>Search Stride (<code>search_stride</code>, default 2; smaller is slower but finer)</label>
      <input type="number" name="stride" min="1" max="32" value="2">

      <label>Optional Known bit_length (leave blank for blind scan)</label>
      <input type="number" name="expected_bit_len" placeholder="e.g. 160 (optional)">

      <div class="inline-check">
        <input type="checkbox" id="multi_scale" name="multi_scale">
        <label for="multi_scale" style="margin:0; font-weight:400;">Enable multi-scale scan (for resized or scaled images)</label>
      </div>

      <button type="submit">Extract Watermark</button>
    </form>

    {% if extract_error %}
      <div class="result-box error">{{ extract_error }}</div>
    {% endif %}

    {% if extract_result is not none %}
      <div class="result-box">
        <strong>Recovered Payload:</strong> <code>{{ extract_result }}</code>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def read_rgb_array(file_storage: Any) -> np.ndarray:
    img = Image.open(file_storage.stream).convert("RGB")
    return np.asarray(img)


def call_extract(
    arr: np.ndarray,
    expected_bit_len: int | None = None,
    search_stride: int = 2,
    multi_scale: bool = False,
) -> Any:
    # Support different extract_from_crop signatures across versions
    kwargs: dict[str, Any] = {"search_stride": search_stride}
    if expected_bit_len is not None:
        kwargs["expected_bit_len"] = expected_bit_len
    if multi_scale:
        kwargs["multi_scale"] = True

    try:
        return pipeline.extract_from_crop(arr, **kwargs)
    except TypeError:
        kwargs.pop("multi_scale", None)
        return pipeline.extract_from_crop(arr, **kwargs)


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    context: dict[str, Any] = {
        "embed_error": None,
        "stego_data_url": None,
        "bit_length": None,
        "autocrop_error": None,
        "autocrop_data_url": None,
        "autocrop_result": None,
        "extract_error": None,
        "extract_result": None,
    }

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "embed":
            file = request.files.get("cover_image")
            payload = request.form.get("payload", "")
            if not file or not file.filename:
                context["embed_error"] = "Please select an image first."
            elif not payload:
                context["embed_error"] = "Please enter a payload to embed."
            else:
                try:
                    arr = read_rgb_array(file)
                    if arr.shape[0] < 64 or arr.shape[1] < 64:
                        context["embed_error"] = "Image dimensions must be at least 64x64 pixels."
                    else:
                        stego_arr, bit_length = pipeline.embed_image(arr, payload)
                        stego_img = Image.fromarray(np.asarray(stego_arr, dtype=np.uint8))
                        context["stego_data_url"] = pil_to_data_url(stego_img, "PNG")
                        context["bit_length"] = bit_length
                except Exception as exc:
                    context["embed_error"] = f"Embed failed: {exc}"

        elif action == "auto_crop_extract":
            file = request.files.get("crop_source_image")
            if not file or not file.filename:
                context["autocrop_error"] = "Please select a watermarked image first."
            else:
                try:
                    ratio = float(request.form.get("crop_ratio", "0.65"))
                    ratio = max(0.2, min(1.0, ratio))
                    bit_len_raw = request.form.get("crop_bit_len", "").strip()
                    expected_bit_len = int(bit_len_raw) if bit_len_raw else None

                    arr = read_rgb_array(file)
                    h, w = arr.shape[0], arr.shape[1]
                    new_h = max(64, int(h * ratio))
                    new_w = max(64, int(w * ratio))
                    top = max(0, (h - new_h) // 2)
                    left = max(0, (w - new_w) // 2)
                    cropped_arr = arr[top : top + new_h, left : left + new_w]

                    cropped_img = Image.fromarray(np.asarray(cropped_arr, dtype=np.uint8))
                    context["autocrop_data_url"] = pil_to_data_url(cropped_img, "PNG")
                    context["autocrop_result"] = call_extract(
                        cropped_arr,
                        expected_bit_len=expected_bit_len,
                        search_stride=2,
                        multi_scale=False,
                    )
                except Exception as exc:
                    context["autocrop_error"] = f"Auto-crop extraction failed: {exc}"

        elif action == "extract":
            file = request.files.get("query_image")
            if not file or not file.filename:
                context["extract_error"] = "Please select an image first."
            else:
                try:
                    stride = max(1, int(request.form.get("stride", "2")))
                    bit_len_raw = request.form.get("expected_bit_len", "").strip()
                    expected_bit_len = int(bit_len_raw) if bit_len_raw else None
                    multi_scale = request.form.get("multi_scale") == "on"

                    arr = read_rgb_array(file)
                    context["extract_result"] = call_extract(
                        arr,
                        expected_bit_len=expected_bit_len,
                        search_stride=stride,
                        multi_scale=multi_scale,
                    )
                except Exception as exc:
                    context["extract_error"] = f"Extraction failed: {exc}"

    return render_template_string(HTML_TEMPLATE, **context)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
