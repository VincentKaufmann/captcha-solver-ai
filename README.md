# captcha-solver-ai

Neural net CAPTCHA solver. Fine-tuned MobileNetV2 + OpenCV. Built for the hell of it.

Takes a reCAPTCHA image grid, splits it into cells, classifies each cell with a custom 19-class MobileNetV2 trained on COCO crops, and tells you which ones match the prompt. Also works live on Playwright pages with stealth patches.

## Install

```bash
pip install captcha-solver-ai
```

With Playwright support (for solving CAPTCHAs on live pages):

```bash
pip install captcha-solver-ai[browser]
playwright install chromium
```

## CLI

```bash
# Solve a CAPTCHA grid image
captcha-solver solve captcha.png --prompt "traffic lights"
captcha-solver solve captcha.png --prompt "buses" --grid 4 --verbose

# Classify any image (shows reCAPTCHA categories, not ImageNet)
captcha-solver classify photo.png --top 5
```

Output:

```
Prompt: traffic lights
Grid:   3x3
Match:  [0, 3, 6]

[X] [ ] [ ]
[X] [ ] [ ]
[X] [ ] [ ]

Found 3 matching cell(s)
```

## Python API

```python
from captcha_solver import CaptchaSolver

solver = CaptchaSolver()

# From an image file
result = solver.solve_file("captcha.png", prompt="Select all images with traffic lights")
print(result.matching_cells)  # [0, 3, 6]
print(result.grid_display())

# From a numpy array (OpenCV)
import cv2
img = cv2.imread("captcha.png")
result = solver.solve(img, prompt="buses", grid_size=3)

# From raw bytes
with open("captcha.png", "rb") as f:
    result = solver.solve_bytes(f.read(), prompt="bicycles")

# Check result
if result.solved:
    print(f"Found matches at cells: {result.matching_cells}")
```

### Lower-level API

```python
from captcha_solver import split_grid, classify_cells, classify_image
import cv2

# Split a grid image into cells
img = cv2.imread("captcha.png")
cells = split_grid(img, grid_size=3)  # returns 9 cell images

# Classify cells against a prompt
results = classify_cells(cells, prompt="traffic lights")
for r in results:
    print(f"Cell {r['index']}: match={r['match']}, confidence={r['target_max_prob']:.1%}")

# Classify a single image (returns reCAPTCHA categories)
preds = classify_image(img, top_k=5)
for class_idx, prob in preds:
    print(f"Class {class_idx}: {prob:.1%}")
```

## Playwright (live CAPTCHA solving)

```python
from playwright.async_api import async_playwright
from captcha_solver import CaptchaSolver

solver = CaptchaSolver()

async with async_playwright() as pw:
    browser = await pw.chromium.launch(headless=True)
    page = await browser.new_page()
    await page.goto("https://some-page-with-recaptcha.com")

    solved = await solver.solve_on_page(page, verbose=True)
    if solved:
        print("CAPTCHA solved!")
```

The `solve_on_page` method handles the full flow: clicks the checkbox with human-like mouse movement, reads the challenge prompt, screenshots the grid, classifies each cell, clicks the matches, hits verify, and retries up to 5 rounds. For unsolvable categories it reloads or skips to get a different challenge.

## v0.2.0 — Custom fine-tuned model

**Before** (v0.1.x): Used a generic ImageNet-1K classifier (1000 classes). Had to map reCAPTCHA keywords to ImageNet class indices. Fire hydrants, crosswalks, stairs had no mapping (0% detection). Model was 13MB and downloaded on first use.

**Now** (v0.2.0): Fine-tuned MobileNetV2 on 11,000+ COCO bounding box crops for 19 reCAPTCHA-specific classes. Model is 8.6MB and ships bundled with the package (no download needed).

### Training details

- **Architecture**: MobileNetV2 backbone (frozen) + custom 19-class classifier head (24,339 trainable params)
- **Training data**: 9,469 train / 2,400 val crops from COCO 2017 bounding box annotations
- **Validation accuracy**: 87.8% (best epoch 10/15)
- **Training time**: ~45 min on CPU
- **Model size**: 8.6 MB ONNX

## Real Google reCAPTCHA benchmarks

Tested live against `google.com/recaptcha/api2/demo` with headless Chromium + stealth patches (navigator.webdriver, fake plugins, chrome.runtime).

### Cell-level classification accuracy (v0.2.0)

| Category | Confidence range | Detection quality | Improvement vs v0.1.x |
|----------|-----------------|-------------------|----------------------|
| Fire hydrants | 44-100% | Excellent | **0% -> 100%** |
| Bicycles | 40-100% | Excellent | Improved |
| Traffic lights | 36-99% | Excellent | Similar |
| Motorcycles | 27-99% | Very good | **0% -> 99%** |
| Buses | 24-95% | Very good | Improved |
| Trucks | 29-75% | Good | Similar |
| Boats | 36-86% | Good | New |
| Trains | 30-93% | Good | New |
| Airplanes | 41-98% | Good | New |
| Parking meters | 23-84% | Good | New |
| Cars | 22-81% | Fair | Vehicle confusion |
| Crosswalks | 0% | No data | Needs training data |
| Stairs | 0% | No data | Needs training data |

### End-to-end solve rate

The end-to-end pass rate on live Google reCAPTCHA is currently low. The model identifies objects correctly but Google's verification is strict:

- **"Please select all matching images"** — edge/partial cells missed (cells where the object is barely visible)
- **"Please try again"** — false positives from similar categories (truck mistaken for bus)
- **"Please also check the new images"** — correctly detected, but replacement tiles need more rounds

Google's reCAPTCHA v2 uses behavioral analysis beyond just image classification (mouse movement patterns, timing, browser fingerprinting), making automated solving inherently difficult.

### Pipeline timing

- Grid split + preprocess: <10ms
- MobileNetV2 inference (9 cells): ~200ms on CPU
- Full solve_on_page round: ~8-12s (mostly human-like delays)

## Supported categories

19 classes in the fine-tuned model:

**11 trained on COCO crops** (strong detection): traffic lights, buses, bicycles, motorcycles, cars, boats, airplanes, trains, trucks, fire hydrants, parking meters

**1 background class**: filters non-target content

**7 without training data yet** (will skip/reload): bridges, mountains, tractors, crosswalks, stairs, palm trees, chimneys

The solver retries up to 5 rounds. If it gets an unsolvable category it clicks reload (3x3) or skip (4x4) to get a different challenge.

## How it works

1. Fine-tuned MobileNetV2 (19 reCAPTCHA classes, 8.6MB ONNX) ships bundled with the package
2. OpenCV splits the CAPTCHA grid into individual cells
3. Each cell is resized to 224x224, ImageNet-normalized, and classified
4. Smart thresholding: if the target class is the top prediction it always matches; partial detections use a lower threshold when the top class is "background"
5. Stealth patches (navigator.webdriver, fake plugins, chrome.runtime) for headless browser anti-detection

No external model download required — everything ships in the pip package.

## Training your own model

```bash
cd training
python -m venv venv && source venv/bin/activate
pip install torch torchvision opencv-python-headless numpy
python train.py
```

This downloads COCO 2017 annotations, extracts bounding box crops for target categories, fine-tunes MobileNetV2, and exports to ONNX. Takes ~45 min on CPU.

## License

MIT
