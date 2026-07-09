# Image/Text Recognition (OCR Pipeline)

**Project 4** of the DecodeLabs AI Industrial Training Kit — the optional
mastery phase. This project integrates a pre-trained computer vision
library into a working recognition pipeline: loading an image, cleaning it
up, extracting text with OCR, and validating the result against a
confidence threshold before trusting it.

## What This Project Does

Implements the OCR path of a machine perception pipeline:

1. **Load** an image (your own file, or an auto-generated sample if none
   is supplied, so the project runs out of the box)
2. **Preprocess** the image with classical computer vision techniques:
   grayscale conversion, Gaussian blur, and Otsu's adaptive thresholding
3. **Recognize** text using Tesseract OCR (via `pytesseract`), extracting
   each word along with its confidence score and bounding box
4. **Filter** results through an 80% confidence gate, so low-confidence
   guesses are rejected rather than presented as fact
5. **Report** the recognized text and save an annotated image with green
   boxes around accepted words and red boxes around rejected ones

## Project Structure

```
image-text-recognition/
├── knowledge_base.py   # Configuration: confidence threshold, PSM modes,
│                        # preprocessing parameters, sample image generator
├── recognizer.py         # The recognition pipeline itself
├── requirements.txt
└── README.md
```

### Why split `knowledge_base.py` from `recognizer.py`?

Same pattern as the earlier projects in this series:

- `knowledge_base.py` — the tunable domain knowledge: what confidence
  threshold counts as "trustworthy," which Tesseract page segmentation
  mode to use, how aggressive the blur/threshold should be. Change the
  numbers here to retune the whole pipeline.
- `recognizer.py` — the fixed processing logic: load, preprocess, OCR,
  filter, annotate, report. This code doesn't change when you retune the
  thresholds above it.

## Getting Started

### Prerequisites
- Python 3.7+
- The Tesseract OCR engine installed on your system (the `pytesseract`
  package is only a Python wrapper — it calls the real `tesseract` binary)

**Install Tesseract:**
```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr

# macOS (Homebrew)
brew install tesseract

# Windows: download the installer from
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Run it

With no arguments, it generates and recognizes a sample image automatically:

```bash
python recognizer.py
```

Or point it at your own image:

```bash
python recognizer.py path/to/your_image.png
```

### Example output

```
============================================================
  Project 4: Image/Text Recognition (OCR Path)
============================================================

No image supplied - generated a sample image: sample_input.png

Recognized text (accepted words only):
  DecodeLabs Training Project 4: Text Recognition Confidence Threshold 80 Percent

Word-level results (10 accepted, 1 rejected):
  [ACCEPT] 'DecodeLabs' - 91.0%
  [ACCEPT] 'Training' - 96.0%
  ...
  [REJECT] 'Al' - 72.0%

============================================================
  Milestone Validation
============================================================
1. Library Integration ....... PASS (pytesseract + OpenCV)
2. Pre-Processing Integrity .. PASS (grayscale + Otsu threshold, cutoff=148)
3. Accuracy Benchmarking ..... PASS (avg accepted confidence=95.0%, threshold=80%)
4. Visual Confirmation ....... PASS (saved to recognition_output.png)
```

`recognition_output.png` is written to the working directory: the
original image with bounding boxes and confidence labels drawn over each
detected word.

## How It Works

### Why preprocess before OCR at all?

To a machine, an image isn't a picture — it's a three-dimensional array of
pixel intensities across height, width, and color channels. Raw images are
usually cluttered with shadows, uneven lighting, and noise, all of which
confuse a text recognizer. Preprocessing strips that clutter out before
OCR ever runs:

1. **Grayscale conversion** collapses the 3-channel color array into a
   single intensity channel — OCR doesn't need color information, only
   contrast between text and background.
2. **Gaussian blur** smooths out small artifacts and sensor noise so a
   handful of stray pixels don't get mistaken for text features.
3. **Otsu's thresholding** automatically calculates the optimal cutoff
   intensity and forces every pixel to pure black or white
   (`pixel >= cutoff -> white, else -> black`), maximizing the contrast
   between characters and background. Unlike a fixed threshold, Otsu's
   method recalculates the cutoff for each image, adapting to its specific
   lighting conditions.

### Why filter by confidence?

An OCR model doesn't "know" what a character is — it calculates the
statistical likelihood of what it might be, and every recognized word
comes with a confidence score reflecting that uncertainty. Displaying
every result as if it were equally certain would mean low-confidence
noise gets treated the same as a clean, high-confidence read. This
project draws a hard line at 80% confidence: below that, a detection is
rejected rather than reported, favoring reliability over recall.

### Tuning recognition with Page Segmentation Modes (PSM)

Tesseract's accuracy depends heavily on telling it what kind of layout to
expect:

| PSM | Use case |
|---|---|
| 3 | Fully automatic segmentation — good default for varied layouts |
| 6 | A single uniform block of text (paragraphs, book pages) |
| 7 | A single line of text (headers, license plates) |
| 11 | Sparse, scattered text (invoices, receipts) |

The default here is PSM 6, tuned for the kind of printed, block-style text
in the sample image. Change `DEFAULT_PSM` in `knowledge_base.py` to match
your own input images.

## Extending the Project

- Point the confidence threshold and PSM mode at real scanned documents
  or photos of signage and compare accuracy
- Add the **Object Detection path** described in the DecodeLabs
  playbook: swap Tesseract for `cv2.dnn` with a MobileNet-SSD model to
  detect and label physical objects instead of text, using the same
  confidence-gating pattern
- Log rejected detections to a file to analyze what a raw threshold
  drops on real-world input
- Batch-process a folder of images instead of a single file

## Roadmap

This project covers **Path 1: OCR** from the DecodeLabs Project 4
playbook. **Path 2: Object Detection** follows the same IPO structure
(preprocessing -> model inference -> confidence filtering -> visual
output) but swaps grayscale/Tesseract for blob construction and a
MobileNet-SSD convolutional network, identifying and locating physical
objects instead of reading text.

## License

This project is licensed under the MIT License.

## Credits

Built as part of the DecodeLabs AI Industrial Training Kit — Batch 2026.
OCR powered by Tesseract (Google) via the `pytesseract` wrapper.
