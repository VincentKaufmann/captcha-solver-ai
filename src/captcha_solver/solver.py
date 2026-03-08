"""
Core CAPTCHA solver engine.

Uses MobileNetV2 (ImageNet classifier) to identify objects in CAPTCHA grid cells
and match them against the challenge prompt.
"""

import os
import urllib.request

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.expanduser("~"), ".captcha_solver")
MOBILENET_ONNX = os.path.join(MODEL_DIR, "mobilenetv2-12.onnx")
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "classification/mobilenet/model/mobilenetv2-12.onnx"
)

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Common reCAPTCHA prompt keywords mapped to ImageNet class indices
# Reference: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
CAPTCHA_CLASS_MAP = {
    "traffic light": [920],
    "bus": [654, 779, 874],
    "bicycle": [444, 671],
    "bike": [444, 671],
    "motorcycle": [670, 665],
    "motorbike": [670, 665],
    "car": [436, 468, 511, 609, 656, 717, 751, 817],
    "taxi": [468],
    "cab": [468],
    "crosswalk": [],
    "bridge": [839],
    "boat": [472, 484, 554, 625, 814, 914],
    "ship": [472, 484, 554, 625, 814, 914],
    "airplane": [404, 405],
    "plane": [404, 405],
    "train": [466, 547, 820, 829],
    "truck": [555, 569, 656, 675, 717, 864, 867],
    "fire hydrant": [],
    "hydrant": [],
    "parking meter": [705],
    "stair": [],
    "mountain": [970, 972, 976, 979, 980],
    "palm": [],
    "chimney": [],
    "tractor": [866],
}


def ensure_model() -> str:
    """Download MobileNetV2 ONNX if not cached. Returns path to model file."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(MOBILENET_ONNX):
        return MOBILENET_ONNX
    print(f"Downloading MobileNetV2 ONNX model to {MOBILENET_ONNX} ...")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "captcha-solver/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with open(MOBILENET_ONNX, "wb") as f:
        f.write(data)
    size_mb = len(data) / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB")
    return MOBILENET_ONNX


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess an image for MobileNetV2: resize, normalize, CHW, batch."""
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return np.expand_dims(img, 0)  # add batch dim


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def _resolve_target_classes(prompt: str) -> set[int]:
    """Map a CAPTCHA prompt string to a set of target ImageNet class indices."""
    prompt_lower = prompt.lower()
    target = set()
    for keyword, indices in CAPTCHA_CLASS_MAP.items():
        if keyword in prompt_lower:
            target.update(indices)
    return target


def split_grid(image: np.ndarray, grid_size: int = 3) -> list[np.ndarray]:
    """Split an image into a grid of cells.

    Args:
        image: Input image as numpy array (BGR or RGB).
        grid_size: Number of rows/columns (3 for 3x3, 4 for 4x4).

    Returns:
        List of cell images in row-major order.
    """
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cells.append(image[y1:y2, x1:x2])
    return cells


def classify_image(
    image: np.ndarray,
    session=None,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Classify a single image using MobileNetV2.

    Args:
        image: Input image as numpy array (BGR).
        session: Optional pre-loaded ONNX session. Created if not provided.
        top_k: Number of top predictions to return.

    Returns:
        List of (class_index, probability) tuples sorted by confidence.
    """
    import onnxruntime as ort

    if session is None:
        model_path = ensure_model()
        session = ort.InferenceSession(model_path)

    input_name = session.get_inputs()[0].name
    preprocessed = _preprocess(image)
    outputs = session.run(None, {input_name: preprocessed})
    probs = _softmax(outputs[0][0])
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(int(idx), float(probs[idx])) for idx in top_indices]


def classify_cells(
    cells: list[np.ndarray],
    prompt: str,
    confidence_threshold: float = 0.05,
    top_k: int = 10,
) -> list[dict]:
    """Classify a list of CAPTCHA grid cells against a prompt.

    Args:
        cells: List of cell images (BGR numpy arrays).
        prompt: The CAPTCHA challenge text (e.g. "Select all images with traffic lights").
        confidence_threshold: Minimum probability for a target class to count as a match.
        top_k: Number of top predictions to check for target class membership.

    Returns:
        List of dicts with keys: index, match (bool), top_prediction (class_index, prob),
        target_max_prob (float).
    """
    import onnxruntime as ort

    model_path = ensure_model()
    session = ort.InferenceSession(model_path)
    target_classes = _resolve_target_classes(prompt)

    results = []
    for i, cell in enumerate(cells):
        preds = classify_image(cell, session=session, top_k=top_k)
        top_indices = {idx for idx, _ in preds}

        # Check if any target class is in top-k
        match = bool(target_classes & top_indices)

        # Also check raw probability of target classes
        input_name = session.get_inputs()[0].name
        preprocessed = _preprocess(cell)
        outputs = session.run(None, {input_name: preprocessed})
        probs = _softmax(outputs[0][0])

        target_max = 0.0
        if target_classes:
            target_probs = [float(probs[idx]) for idx in target_classes if idx < len(probs)]
            target_max = max(target_probs) if target_probs else 0.0
            if target_max > confidence_threshold:
                match = True

        results.append({
            "index": i,
            "match": match,
            "top_prediction": preds[0] if preds else (0, 0.0),
            "target_max_prob": target_max,
        })

    return results


class CaptchaSolver:
    """High-level CAPTCHA solver.

    Usage:
        solver = CaptchaSolver()

        # From a grid image
        solution = solver.solve(image, prompt="Select all images with traffic lights")
        print(solution.matching_cells)  # [0, 3, 6]

        # From a grid image file
        solution = solver.solve_file("captcha.png", prompt="buses", grid_size=3)

        # With Playwright (requires `pip install captcha-solver-ai[browser]`)
        solved = await solver.solve_on_page(playwright_page)
    """

    def __init__(self):
        self._session = None

    def _get_session(self):
        if self._session is None:
            import onnxruntime as ort
            model_path = ensure_model()
            self._session = ort.InferenceSession(model_path)
        return self._session

    def solve(
        self,
        grid_image: np.ndarray,
        prompt: str,
        grid_size: int = 3,
        confidence_threshold: float = 0.05,
    ) -> "SolveResult":
        """Solve a CAPTCHA grid image.

        Args:
            grid_image: The full CAPTCHA grid image (BGR numpy array).
            prompt: Challenge text like "Select all images with traffic lights".
            grid_size: 3 for 3x3 grid, 4 for 4x4 grid.
            confidence_threshold: Minimum probability for a match.

        Returns:
            SolveResult with matching cell indices and details.
        """
        cells = split_grid(grid_image, grid_size)
        results = classify_cells(cells, prompt, confidence_threshold)
        matching = [r["index"] for r in results if r["match"]]
        return SolveResult(
            matching_cells=matching,
            grid_size=grid_size,
            prompt=prompt,
            cell_details=results,
        )

    def solve_file(
        self,
        image_path: str,
        prompt: str,
        grid_size: int = 3,
        confidence_threshold: float = 0.05,
    ) -> "SolveResult":
        """Solve a CAPTCHA from an image file.

        Args:
            image_path: Path to the CAPTCHA grid image.
            prompt: Challenge text.
            grid_size: 3 or 4.
            confidence_threshold: Minimum probability for a match.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.solve(img, prompt, grid_size, confidence_threshold)

    def solve_bytes(
        self,
        image_bytes: bytes,
        prompt: str,
        grid_size: int = 3,
        confidence_threshold: float = 0.05,
    ) -> "SolveResult":
        """Solve a CAPTCHA from raw image bytes (PNG/JPEG)."""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return self.solve(img, prompt, grid_size, confidence_threshold)

    async def solve_on_page(self, page, max_rounds: int = 3) -> bool:
        """Attempt to solve a reCAPTCHA on a live Playwright page.

        Requires: pip install captcha-solver-ai[browser]

        Args:
            page: A Playwright page object with a visible reCAPTCHA.
            max_rounds: Maximum number of challenge rounds to attempt.

        Returns:
            True if the CAPTCHA was solved, False otherwise.
        """
        import random

        try:
            # Step 1: Click the checkbox in the anchor iframe (not bframe)
            anchor_frame = page.frame_locator("iframe[src*='anchor']")
            checkbox = anchor_frame.locator(
                "#recaptcha-anchor, .recaptcha-checkbox-border"
            )
            if await checkbox.count() > 0:
                box = await checkbox.first.bounding_box()
                if box:
                    x = box["x"] + box["width"] * random.uniform(0.3, 0.7)
                    y = box["y"] + box["height"] * random.uniform(0.3, 0.7)
                    await page.mouse.move(
                        x - random.randint(50, 150), y - random.randint(50, 150)
                    )
                    await page.wait_for_timeout(random.randint(100, 300))
                    await page.mouse.move(x, y, steps=random.randint(10, 25))
                    await page.wait_for_timeout(random.randint(200, 500))
                    await page.mouse.click(x, y)
                    await page.wait_for_timeout(random.randint(3000, 5000))

                    # Check if checkbox alone was enough (green checkmark)
                    try:
                        checked = anchor_frame.locator(
                            ".recaptcha-checkbox-checked, "
                            "[aria-checked='true']"
                        )
                        if await checked.count() > 0:
                            return True
                    except Exception:
                        pass

            # Step 2: Solve image challenges (may take multiple rounds)
            for _round in range(max_rounds):
                challenge_frame = None
                for frame in page.frames:
                    url = frame.url or ""
                    if "recaptcha" in url and "bframe" in url:
                        challenge_frame = frame
                        break

                if not challenge_frame:
                    return False

                # Read the prompt
                prompt_el = challenge_frame.locator(
                    ".rc-imageselect-desc-no-canonical, .rc-imageselect-desc, "
                    ".rc-imageselect-instructions"
                )
                if await prompt_el.count() == 0:
                    return False

                prompt_text = await prompt_el.first.inner_text()

                # Screenshot the grid
                grid = challenge_frame.locator(
                    "table[class*='rc-imageselect-table'], "
                    ".rc-imageselect-target"
                )
                if await grid.count() == 0:
                    return False

                grid_screenshot = await grid.first.screenshot()
                if not grid_screenshot:
                    return False

                # Detect grid size
                is_4x4 = await challenge_frame.locator(
                    ".rc-imageselect-table-44"
                ).count()
                grid_size = 4 if is_4x4 > 0 else 3

                # Get tile cells
                tiles = challenge_frame.locator(
                    "table[class*='rc-imageselect-table'] td"
                )
                tile_count = await tiles.count()

                # Solve it
                result = self.solve_bytes(
                    grid_screenshot, prompt_text, grid_size
                )

                if not result.matching_cells:
                    # No matches — click skip if available, otherwise fail
                    skip_btn = challenge_frame.locator(
                        "#recaptcha-verify-button"
                    )
                    btn_text = ""
                    if await skip_btn.count() > 0:
                        btn_text = (
                            await skip_btn.first.inner_text()
                        ).strip().lower()
                    if "skip" in btn_text:
                        await skip_btn.first.click()
                        await page.wait_for_timeout(
                            random.randint(2000, 4000)
                        )
                        continue
                    return False

                # Click matching cells with human timing
                for cell_idx in result.matching_cells:
                    if cell_idx < tile_count:
                        tile = tiles.nth(cell_idx)
                        box = await tile.bounding_box()
                        if box:
                            x = box["x"] + box["width"] * random.uniform(
                                0.3, 0.7
                            )
                            y = box["y"] + box["height"] * random.uniform(
                                0.3, 0.7
                            )
                            await page.mouse.click(x, y)
                            await page.wait_for_timeout(
                                random.randint(300, 700)
                            )

                await page.wait_for_timeout(random.randint(1500, 3000))

                # Click verify
                verify_btn = challenge_frame.locator(
                    "#recaptcha-verify-button"
                )
                if await verify_btn.count() > 0:
                    await verify_btn.first.click()
                    await page.wait_for_timeout(random.randint(3000, 5000))

                # Check if solved
                try:
                    checked = anchor_frame.locator(
                        ".recaptcha-checkbox-checked, "
                        "[aria-checked='true']"
                    )
                    if await checked.count() > 0:
                        return True
                except Exception:
                    pass

                # Check if challenge is still showing
                still_blocked = await page.locator(
                    "iframe[src*='recaptcha'][src*='bframe']"
                ).count()
                if still_blocked == 0:
                    return True

            return False

        except Exception:
            return False


class SolveResult:
    """Result of a CAPTCHA solve attempt."""

    def __init__(self, matching_cells, grid_size, prompt, cell_details):
        self.matching_cells = matching_cells
        self.grid_size = grid_size
        self.prompt = prompt
        self.cell_details = cell_details

    @property
    def solved(self) -> bool:
        return len(self.matching_cells) > 0

    def grid_display(self) -> str:
        """Return an ASCII grid showing which cells matched."""
        lines = []
        for row in range(self.grid_size):
            row_str = ""
            for col in range(self.grid_size):
                idx = row * self.grid_size + col
                row_str += " [X]" if idx in self.matching_cells else " [ ]"
            lines.append(row_str.strip())
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"SolveResult(prompt={self.prompt!r}, "
            f"matching={self.matching_cells}, "
            f"grid={self.grid_size}x{self.grid_size})"
        )
