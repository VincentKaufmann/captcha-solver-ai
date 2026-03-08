"""
CLI for captcha-solver-ai.

Usage:
    captcha-solver solve captcha.png --prompt "traffic lights"
    captcha-solver solve captcha.png --prompt "buses" --grid 4
    captcha-solver classify image.png --top 5
    captcha-solver download-model
"""

import argparse
import sys

import cv2


def cmd_solve(args):
    from captcha_solver.solver import CaptchaSolver

    solver = CaptchaSolver()
    result = solver.solve_file(
        args.image,
        prompt=args.prompt,
        grid_size=args.grid,
        confidence_threshold=args.threshold,
    )

    print(f"Prompt: {result.prompt}")
    print(f"Grid:   {result.grid_size}x{result.grid_size}")
    print(f"Match:  {result.matching_cells if result.matching_cells else 'none'}")
    print()
    print(result.grid_display())
    print()

    if result.solved:
        print(f"Found {len(result.matching_cells)} matching cell(s)")
    else:
        print("No matching cells found")

    if args.verbose:
        print("\nCell details:")
        for d in result.cell_details:
            cls_idx, cls_prob = d["top_prediction"]
            mark = "*" if d["match"] else " "
            print(
                f"  [{mark}] Cell {d['index']:2d}: "
                f"top class={cls_idx} ({cls_prob:.1%}), "
                f"target prob={d['target_max_prob']:.1%}"
            )


def cmd_classify(args):
    from captcha_solver.solver import classify_image, ensure_model

    import onnxruntime as ort

    model_path = ensure_model()
    session = ort.InferenceSession(model_path)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read {args.image}", file=sys.stderr)
        sys.exit(1)

    preds = classify_image(img, session=session, top_k=args.top)

    print(f"Top {args.top} predictions for {args.image}:")
    for idx, prob in preds:
        print(f"  Class {idx:4d}: {prob:.1%}")


def cmd_download(args):
    from captcha_solver.solver import ensure_model

    path = ensure_model()
    print(f"Model ready at {path}")


def main():
    parser = argparse.ArgumentParser(
        prog="captcha-solver",
        description="Neural net CAPTCHA solver using MobileNetV2 + OpenCV",
    )
    sub = parser.add_subparsers(dest="command")

    # solve
    p_solve = sub.add_parser("solve", help="Solve a CAPTCHA grid image")
    p_solve.add_argument("image", help="Path to CAPTCHA grid image")
    p_solve.add_argument(
        "--prompt", "-p", required=True,
        help="Challenge text (e.g. 'traffic lights', 'buses')",
    )
    p_solve.add_argument(
        "--grid", "-g", type=int, default=3, choices=[3, 4],
        help="Grid size (3 for 3x3, 4 for 4x4, default: 3)",
    )
    p_solve.add_argument(
        "--threshold", "-t", type=float, default=0.05,
        help="Confidence threshold (default: 0.05)",
    )
    p_solve.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-cell classification details",
    )

    # classify
    p_cls = sub.add_parser("classify", help="Classify a single image")
    p_cls.add_argument("image", help="Path to image")
    p_cls.add_argument(
        "--top", "-k", type=int, default=5,
        help="Number of top predictions (default: 5)",
    )

    # download-model
    sub.add_parser("download-model", help="Pre-download the MobileNetV2 model")

    args = parser.parse_args()

    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "download-model":
        cmd_download(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
