"""
knowledge_base.py
------------------
The "domain knowledge" layer for the Image/Text Recognition project.

Holds the configuration values a recognition pipeline needs (confidence
threshold, Tesseract page segmentation modes, preprocessing parameters)
plus a helper to generate a sample test image, so the project can be run
and verified without requiring the user to supply their own image.
"""

from PIL import Image, ImageDraw, ImageFont
import os

# ---------------------------------------------------------------------------
# RECOGNITION CONFIGURATION
# ---------------------------------------------------------------------------

# Minimum confidence (0-100) a recognized word must have to be accepted.
# This matches the project's "80% Confidence Gate": results below this
# threshold are treated as unreliable guesses and dropped rather than
# displayed as if they were certain.
CONFIDENCE_THRESHOLD = 80

# Tesseract Page Segmentation Modes (PSM) and when to use each one.
# Passed to pytesseract as `--psm N` in the config string.
PSM_MODES = {
    3: "Fully automatic page segmentation (default, good for varied layouts).",
    6: "Assume a single uniform block of text (e.g. a book page or paragraph).",
    7: "Treat the image as a single text line (e.g. a header or a plate number).",
    11: "Sparse text: find as much text as possible in no particular order "
        "(e.g. an invoice or receipt with scattered fields).",
}

DEFAULT_PSM = 6

# Adaptive thresholding parameters (see preprocessing.py). These control
# how aggressively local contrast is normalized before OCR.
ADAPTIVE_BLOCK_SIZE = 31   # size of the local neighborhood (must be odd)
ADAPTIVE_C = 15            # constant subtracted from the local mean
GAUSSIAN_BLUR_KERNEL = (5, 5)


# ---------------------------------------------------------------------------
# SAMPLE DATA GENERATION
# Lets the project be demoed end-to-end without needing an external image.
# ---------------------------------------------------------------------------

SAMPLE_TEXT_LINES = [
    "DecodeLabs AI Training",
    "Project 4: Text Recognition",
    "Confidence Threshold 80 Percent",
]


def generate_sample_image(output_path: str = "sample_input.png") -> str:
    """
    Generate a synthetic test image containing printed text, with mild
    noise added to simulate a real-world scan (so the preprocessing step
    has something meaningful to clean up).

    Returns the path to the generated image.
    """
    width, height = 800, 300
    image = Image.new("RGB", (width, height), color=(235, 235, 230))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
        )
    except OSError:
        font = ImageFont.load_default()

    y = 40
    for line in SAMPLE_TEXT_LINES:
        draw.text((40, y), line, fill=(20, 20, 20), font=font)
        y += 60

    # Light synthetic noise so grayscale + adaptive thresholding has a
    # real job to do, similar to shadows/scan artifacts on a real photo.
    import numpy as np
    arr = np.array(image).astype("int16")
    noise = np.random.randint(-20, 20, arr.shape, dtype="int16")
    arr = (arr + noise).clip(0, 255).astype("uint8")
    noisy_image = Image.fromarray(arr)

    noisy_image.save(output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    # Quick manual check: python knowledge_base.py
    path = generate_sample_image()
    print(f"Sample image generated at: {path}")
    print(f"\nConfidence threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"Default PSM mode: {DEFAULT_PSM} - {PSM_MODES[DEFAULT_PSM]}")
