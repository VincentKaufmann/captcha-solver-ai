"""
captcha-solver-ai: Neural net CAPTCHA solver using MobileNetV2 + OpenCV.

Built for the hell of it.
"""

from captcha_solver.solver import (
    CaptchaSolver,
    classify_cells,
    classify_image,
    split_grid,
)

__all__ = ["CaptchaSolver", "classify_cells", "classify_image", "split_grid"]
__version__ = "0.2.0"
