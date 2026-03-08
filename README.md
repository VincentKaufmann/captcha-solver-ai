# captcha-solver-ai

Neural net CAPTCHA solver. MobileNetV2 + OpenCV. Built for the hell of it.

Takes a reCAPTCHA image grid, splits it into cells, classifies each cell with a MobileNetV2 ImageNet classifier, and tells you which ones match the prompt. Also works live on Playwright pages.

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

# Classify any image
captcha-solver classify photo.png --top 10

# Pre-download the model (~13MB, auto-downloads on first use anyway)
captcha-solver download-model
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

# Classify a single image (raw ImageNet predictions)
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

    solved = await solver.solve_on_page(page)
    if solved:
        print("CAPTCHA solved!")
```

The `solve_on_page` method handles the full flow: clicks the checkbox with human-like mouse movement, and if Google serves an image challenge it screenshots the grid, classifies each cell, clicks the matches, and hits verify.

## Real Google reCAPTCHA benchmarks

Tested live against `google.com/recaptcha/api2/demo` with headless Chromium + stealth patches.

| Challenge | Grid | Result | Top confidence | Notes |
|-----------|------|--------|----------------|-------|
| Buses | 3x3 | 3/3 cells correct | 26.8% (minibus) | Cells 5,6,7 matched |
| Traffic lights | 4x4 | 1/1 detected | 98.0% (class 920) | Partial cells missed on 4x4 |
| Motorcycles | 3x3 | Detected | - | Served after round 1 |
| Bicycles | 3x3 | Detected | - | Served after round 1 |
| Crosswalks | 3x3 | Skip | 0% | No ImageNet class exists |
| Fire hydrants | 3x3 | Skip | 0% | No ImageNet class exists |

**Pipeline timing** (single 3x3 grid, 9 cells):
- Grid split + preprocess: <10ms
- MobileNetV2 inference (9 cells): ~200ms on CPU
- Full solve_on_page round: ~8-12s (mostly waiting on human-like delays)

**What works well**: buses, traffic lights, cars, trucks, trains, airplanes, motorcycles, bicycles, boats, bridges, tractors

**What doesn't work**: crosswalks, fire hydrants, stairs, palm trees, chimneys have no ImageNet-1K equivalent. The solver skips these and moves to the next round.

4x4 grids are harder than 3x3 because each cell is smaller and may only show a partial object.

## Supported CAPTCHA categories

13 categories with working ImageNet class mappings:

traffic lights, buses, bicycles, motorcycles, cars, taxis, bridges, boats, ships, airplanes, trains, trucks, parking meters, mountains, tractors

9 categories without ImageNet equivalents (will skip):

crosswalks, fire hydrants, stairs, palm trees, chimneys, cabs (duplicate of taxi)

The solver retries up to 3 rounds. If it gets an unsolvable category it skips and tries the next one.

## How it works

1. MobileNetV2 (pre-trained on ImageNet, 1000 classes) runs as an ONNX model (~13MB)
2. OpenCV splits the CAPTCHA grid into individual cells
3. Each cell is resized to 224x224, normalized, and fed through the network
4. Top-10 predictions are checked against a mapping of CAPTCHA keywords to ImageNet class indices
5. Cells where a target class appears in the top-10 or exceeds 5% probability are marked as matches

The model auto-downloads on first use and is cached at `~/.captcha_solver/`.

## License

MIT
