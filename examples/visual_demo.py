"""
visual_demo.py - Interactive Visual Inspector for spectral-watermark
Saves a 4-panel diagnostic figure to 'demo_output.png'.
"""

import math
import subprocess
import sys
import numpy as np
import matplotlib
# Use headless backend to ensure compatibility on any system
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectral_watermark import WatermarkPipeline


def generate_sample_canvas(size: int = 256) -> np.ndarray:
    """Generates a realistic test image canvas with natural texture."""
    rng = np.random.default_rng(2026)
    # Natural spatial gradient + high/mid frequency texture
    y = np.linspace(60, 200, size).reshape(-1, 1)
    x = np.linspace(40, 180, size).reshape(1, -1)
    gradient = (y + x) / 2.0
    texture = rng.normal(0, 20.0, size=(size, size))
    return np.clip(gradient + texture, 0, 255).astype(np.uint8)



def compute_psnr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((ref.astype(np.float64) - deg.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def main() -> None:
    print("[*] Initializing spectral watermark pipeline...")
    pipeline = WatermarkPipeline(block_size=64)
    secret_text = "TOP_SECRET_IOAI"
    original = generate_sample_canvas(256)

    # 1. Embed Secret Watermark
    stego, bit_len = pipeline.embed_image(original, secret_text)
    psnr_val = compute_psnr(original, stego)
    print(f"[*] Watermark embedded successfully (PSNR: {psnr_val:.2f} dB)")

    # 2. Amplify Residual (Difference) to display frequency modifications
    residual = np.abs(stego.astype(np.float64) - original.astype(np.float64))
    amplified_residual = np.clip(residual * 20.0, 0, 255).astype(np.uint8)



    # 3. Simulate a 40% Cropping Attack
    crop_h, crop_w = 160, 160
    off_y, off_x = 45, 35
    cropped = stego[off_y : off_y + crop_h, off_x : off_x + crop_w]
    print(f"[*] Applied 40% cropping attack (Canvas size: {crop_w}x{crop_h} px)")

    # 4. Extract Payload from Cropped Image
    print("[*] Performing blind sliding-window extraction...")
    recovered_text = pipeline.extract_from_crop(cropped, expected_bit_len=bit_len, search_stride=2)
    print(f"[*] Recovered payload: '{recovered_text}'")

    # 5. Render & Save 4-Panel Visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(f"spectral-watermark Visual Inspector | Recovered: '{recovered_text}'", fontsize=14, fontweight="bold")

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("1. Original Image (256x256)")
    axes[0].axis("off")

    axes[1].imshow(stego, cmap="gray")
    axes[1].set_title(f"2. Stego Image\n(PSNR: {psnr_val:.1f} dB)")
    axes[1].axis("off")

    axes[2].imshow(amplified_residual, cmap="inferno")
    axes[2].set_title("3. Residual (Diff x 20)\nMid-Freq Rings")
    axes[2].axis("off")

    axes[3].imshow(cropped, cmap="gray")
    axes[3].set_title(f"4. 40% Crop Attack (160x160)\nPayload: '{recovered_text}'")
    axes[3].axis("off")

    plt.tight_layout()
    output_filename = "demo_output.png"
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[+] Visualization saved successfully to: {output_filename}")


if __name__ == "__main__":
    main()
