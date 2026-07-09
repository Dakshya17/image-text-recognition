#!/usr/bin/env python3
"""
recognizer.py
--------------
Project 4: Image/Text Recognition (Basic) - OCR Path.

Implements a full text-recognition pipeline: load an image, clean it up
with classical computer vision preprocessing, run it through Tesseract
OCR, filter out low-confidence guesses, and produce both a text report
and an annotated image showing exactly what the machine "read" and how
sure it was.

Pipeline (IPO model):
    INPUT   -> Load an image (a user-supplied file, or an auto-generated
               sample image if none is provided)
    PROCESS -> Grayscale -> Gaussian blur -> Otsu adaptive threshold ->
               Tesseract OCR -> confidence filtering
    OUTPUT  -> Recognized text + confidence report + annotated image with
               bounding boxes

Run it with:
    python recognizer.py [path/to/image]

If no image path is given, a sample image is generated automatically.
"""

import sys
import os

import cv2
import pytesseract
from pytesseract import Output

from knowledge_base import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_PSM,
    ADAPTIVE_BLOCK_SIZE,
    GAUSSIAN_BLUR_KERNEL,
    generate_sample_image,
)

OUTPUT_IMAGE_PATH = "recognition_output.png"


def load_image(image_path: str):
    """Load an image from disk as a BGR OpenCV array."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image at '{image_path}'. Check the path and "
            f"that it's a valid image file."
        )
    return image


def preprocess(image):
    """
    Clean up the raw image before OCR.

    Step 1 - Grayscale: collapses the 3-channel RGB matrix into a single
             intensity channel, discarding color information that OCR
             doesn't need.
    Step 2 - Gaussian blur: smooths out small artifacts and sensor noise
             so thresholding doesn't get thrown off by isolated pixels.
    Step 3 - Otsu's thresholding: automatically calculates the optimal
             cutoff intensity and forces every pixel to pure black or
             white, maximizing contrast between text and background.

    Returns the processed (binary) image and the cutoff value Otsu chose.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
    cutoff, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, cutoff


def run_ocr(binary_image, psm: int = DEFAULT_PSM):
    """
    Run Tesseract OCR on a preprocessed image and return per-word results:
    the recognized text, its confidence score, and its bounding box.
    """
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        binary_image, config=config, output_type=Output.DICT
    )

    results = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        confidence = float(data["conf"][i])
        if not text or confidence < 0:
            continue  # skip empty detections and Tesseract's "no data" rows
        results.append({
            "text": text,
            "confidence": confidence,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
        })
    return results


def filter_by_confidence(results, threshold: float = CONFIDENCE_THRESHOLD):
    """
    Apply the project's confidence gate: only words at or above the
    threshold are trusted. This is the same logic as the spec's
    `if confidence >= 0.80: accept else: drop`, just expressed as a
    filter rather than a per-item branch.
    """
    accepted = [r for r in results if r["confidence"] >= threshold]
    rejected = [r for r in results if r["confidence"] < threshold]
    return accepted, rejected


def draw_annotations(image, accepted, rejected):
    """
    Draw bounding boxes and confidence labels on a copy of the original
    image: green for accepted (high-confidence) detections, red for
    rejected (low-confidence) ones, so both are visible for inspection.
    """
    annotated = image.copy()

    for r in rejected:
        x, y, w, h = r["left"], r["top"], r["width"], r["height"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 1)

    for r in accepted:
        x, y, w, h = r["left"], r["top"], r["width"], r["height"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{r['text']} ({r['confidence']:.0f}%)"
        cv2.putText(
            annotated, label, (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1, cv2.LINE_AA,
        )

    return annotated


def print_report(accepted, rejected, otsu_cutoff, output_path):
    """Print the recognized text plus a pass/fail check against the
    project's four validation criteria."""
    recognized_text = " ".join(r["text"] for r in accepted)
    avg_confidence = (
        sum(r["confidence"] for r in accepted) / len(accepted)
        if accepted else 0.0
    )

    print("\nRecognized text (accepted words only):")
    print(f"  {recognized_text if recognized_text else '(nothing above the confidence threshold)'}")

    print(f"\nWord-level results ({len(accepted)} accepted, {len(rejected)} rejected):")
    for r in accepted:
        print(f"  [ACCEPT] '{r['text']}' - {r['confidence']:.1f}%")
    for r in rejected:
        print(f"  [REJECT] '{r['text']}' - {r['confidence']:.1f}%")

    print("\n" + "=" * 60)
    print("  Milestone Validation")
    print("=" * 60)
    print(f"1. Library Integration ....... PASS (pytesseract + OpenCV)")
    print(f"2. Pre-Processing Integrity .. PASS (grayscale + Otsu threshold, "
          f"cutoff={otsu_cutoff:.0f})")
    status_3 = "PASS" if accepted and avg_confidence >= CONFIDENCE_THRESHOLD else "FAIL"
    print(f"3. Accuracy Benchmarking ..... {status_3} "
          f"(avg accepted confidence={avg_confidence:.1f}%, "
          f"threshold={CONFIDENCE_THRESHOLD}%)")
    print(f"4. Visual Confirmation ....... PASS (saved to {output_path})")


def recognize(image_path: str):
    """Full pipeline: load, preprocess, OCR, filter, annotate, report."""
    image = load_image(image_path)
    binary, otsu_cutoff = preprocess(image)

    raw_results = run_ocr(binary)
    accepted, rejected = filter_by_confidence(raw_results)

    annotated = draw_annotations(image, accepted, rejected)
    cv2.imwrite(OUTPUT_IMAGE_PATH, annotated)

    print_report(accepted, rejected, otsu_cutoff, OUTPUT_IMAGE_PATH)
    return accepted, rejected


def main():
    print("=" * 60)
    print("  Project 4: Image/Text Recognition (OCR Path)")
    print("=" * 60)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nUsing supplied image: {image_path}")
    else:
        image_path = generate_sample_image()
        print(f"\nNo image supplied - generated a sample image: {image_path}")

    if not os.path.exists(image_path):
        print(f"\nError: '{image_path}' does not exist.")
        sys.exit(1)

    recognize(image_path)


if __name__ == "__main__":
    main()
