"""
CLI orchestrator for the OCR synthetic document toolkit.

End-to-end pipeline: generate content → render PDF → degrade image →
inject text errors → run OCR → compute metrics.

Usage:
    python pipeline.py --doc-type invoice --count 10 --output-dir ./output
    python pipeline.py --doc-type all --count 5 --error-levels 0.3,1.0,3.0,7.0
    python pipeline.py --skip-api --input-json content.json --doc-type invoice
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from confusions import (
    compute_cer,
    compute_wer,
    inject_errors,
)
from templates import (
    degrade_image,
    pdf_to_images,
    render_document,
)
from content import generate_batch, GENERATORS

# Optional: Tesseract for OCR evaluation
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


DEFAULT_ERROR_LEVELS = [0.3, 1.0, 3.0, 7.0, 11.0, 15.0]


def run_ocr(image) -> str:
    """Run Tesseract OCR on a PIL Image and return the recognized text."""
    if not HAS_TESSERACT:
        return "[OCR skipped: pytesseract not available]"
    return pytesseract.image_to_string(image)


def process_document(
    doc_data: dict[str, Any],
    doc_type: str,
    doc_index: int,
    output_dir: Path,
    error_levels: list[float],
    degradation_params: dict[str, Any],
    no_ocr: bool = False,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Process a single document through the full pipeline.

    Returns a summary dict with metrics for each error level.
    """
    doc_name = f"{doc_type}_{doc_index:03d}"
    doc_dir = output_dir / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Render PDF + extract ground truth
    pdf_bytes, ground_truth = render_document(doc_type, doc_data)
    (doc_dir / "clean.pdf").write_bytes(pdf_bytes)
    (doc_dir / "ground_truth.txt").write_text(ground_truth, encoding="utf-8")

    # Save the content JSON for reproducibility
    (doc_dir / "content.json").write_text(
        json.dumps(doc_data, indent=2), encoding="utf-8"
    )

    # Step 2: Convert PDF to image + degrade
    try:
        images = pdf_to_images(pdf_bytes)
        base_image = images[0]  # Use first page
    except (ImportError, Exception) as e:
        print(f"    Warning: PDF-to-image failed ({e}). Skipping image pipeline.")
        base_image = None

    degraded_image = None
    if base_image is not None:
        deg_seed = (seed * 1000 + doc_index) if seed is not None else None
        degraded_image = degrade_image(base_image, seed=deg_seed, **degradation_params)
        degraded_image.save(str(doc_dir / "degraded.png"), format="PNG")

    # Step 3: Per error-level processing
    level_results = []
    for level in error_levels:
        level_name = f"level_{level}"
        level_dir = doc_dir / level_name
        level_dir.mkdir(exist_ok=True)

        # Inject text errors at this level
        level_seed = (seed * 1000 + doc_index * 100 + int(level * 10)) if seed is not None else None
        noisy_text = inject_errors(ground_truth, error_level=level, seed=level_seed)
        (level_dir / "noisy_text.txt").write_text(noisy_text, encoding="utf-8")

        # Save degraded image in each level dir for convenience
        if degraded_image is not None:
            degraded_image.save(str(level_dir / "degraded.png"), format="PNG")

        # Run OCR on degraded image
        ocr_text = ""
        if not no_ocr and degraded_image is not None:
            ocr_text = run_ocr(degraded_image)
            (level_dir / "ocr_output.txt").write_text(ocr_text, encoding="utf-8")

        # Compute metrics
        text_cer = compute_cer(ground_truth, noisy_text)
        text_wer = compute_wer(ground_truth, noisy_text)

        metrics = {
            "error_level_target": level,
            "text_injection_cer": round(text_cer * 100, 2),
            "text_injection_wer": round(text_wer * 100, 2),
        }

        if ocr_text and ocr_text != "[OCR skipped: pytesseract not available]":
            ocr_cer = compute_cer(ground_truth, ocr_text)
            ocr_wer = compute_wer(ground_truth, ocr_text)
            metrics["ocr_cer"] = round(ocr_cer * 100, 2)
            metrics["ocr_wer"] = round(ocr_wer * 100, 2)

        (level_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        level_results.append(metrics)

    return {
        "document": doc_name,
        "doc_type": doc_type,
        "ground_truth_length": len(ground_truth),
        "levels": level_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="OCR Synthetic Document Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --doc-type invoice --count 10 --output-dir ./output
  python pipeline.py --doc-type all --count 5 --error-levels 0.3,1.0,3.0,7.0
  python pipeline.py --skip-api --input-json content.json --doc-type invoice
        """,
    )

    # Core options
    parser.add_argument(
        "--doc-type", default="invoice",
        choices=list(GENERATORS.keys()) + ["all"],
        help="Document type to generate (default: invoice)",
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of documents to generate per type (default: 5)",
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--error-levels", default=None,
        help="Comma-separated CER targets (default: 0.3,1.0,3.0,7.0,11.0,15.0)",
    )

    # API options
    parser.add_argument(
        "--model", default="claude-sonnet-4-20250514",
        help="Claude model for content generation",
    )
    parser.add_argument(
        "--skip-api", action="store_true",
        help="Skip API calls; load content from --input-json instead",
    )
    parser.add_argument(
        "--input-json", default=None,
        help="Path to JSON file with pre-generated content (use with --skip-api)",
    )

    # Degradation parameters
    parser.add_argument("--blur", type=float, default=0.8, help="Blur radius (default: 0.8)")
    parser.add_argument("--noise", type=float, default=8.0, help="Noise sigma (default: 8.0)")
    parser.add_argument("--rotation", type=float, default=0.3, help="Rotation degrees (default: 0.3)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality (default: 80)")
    parser.add_argument("--ink-bleed", type=float, default=0.0, help="Ink bleed amount (default: 0.0)")

    # Other options
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-ocr", action="store_true", help="Skip Tesseract OCR step")

    args = parser.parse_args()

    # Parse error levels
    if args.error_levels:
        error_levels = [float(x.strip()) for x in args.error_levels.split(",")]
    else:
        error_levels = DEFAULT_ERROR_LEVELS

    # Determine document types to process
    if args.doc_type == "all":
        doc_types = list(GENERATORS.keys())
    else:
        doc_types = [args.doc_type]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    degradation_params = {
        "blur": args.blur,
        "noise": args.noise,
        "rotation": args.rotation,
        "jpeg_quality": args.jpeg_quality,
        "ink_bleed": args.ink_bleed,
    }

    all_summaries: list[dict] = []

    for doc_type in doc_types:
        print(f"\n{'='*60}")
        print(f"Processing document type: {doc_type}")
        print(f"{'='*60}")

        # Step 1: Get content (API or JSON file)
        if args.skip_api:
            if args.input_json is None:
                print("Error: --skip-api requires --input-json", file=sys.stderr)
                sys.exit(1)
            with open(args.input_json, "r") as f:
                all_content = json.load(f)
            # Handle both list and dict-of-lists formats
            if isinstance(all_content, dict):
                content_list = all_content.get(doc_type, [])
                if not content_list:
                    print(f"  Warning: no '{doc_type}' key in {args.input_json}, skipping",
                          file=sys.stderr)
                    continue
            else:
                content_list = all_content
            if len(content_list) < args.count:
                print(f"  Warning: JSON has {len(content_list)} items but "
                      f"--count is {args.count}; processing {len(content_list)}",
                      file=sys.stderr)
            content_list = content_list[:args.count]
        else:
            print(f"  Generating {args.count} {doc_type} document(s) via API...")
            content_list = generate_batch(
                doc_type=doc_type,
                count=args.count,
                model=args.model,
            )
            # Save generated content for reproducibility
            content_file = output_dir / f"{doc_type}_content.json"
            content_file.write_text(
                json.dumps(content_list, indent=2), encoding="utf-8"
            )
            print(f"  Saved content to {content_file}")

        # Step 2: Process each document
        for i, doc_data in enumerate(content_list, 1):
            print(f"\n  Document {i}/{len(content_list)}:")
            print(f"    Rendering PDF...")
            summary = process_document(
                doc_data=doc_data,
                doc_type=doc_type,
                doc_index=i,
                output_dir=output_dir,
                error_levels=error_levels,
                degradation_params=degradation_params,
                no_ocr=args.no_ocr,
                seed=args.seed,
            )
            all_summaries.append(summary)

            # Print quick metrics
            for lev in summary["levels"]:
                print(f"    Level {lev['error_level_target']}%: "
                      f"CER={lev['text_injection_cer']}% "
                      f"WER={lev['text_injection_wer']}%"
                      + (f" OCR_CER={lev.get('ocr_cer', 'N/A')}%" if 'ocr_cer' in lev else ""))

    # Write summary
    summary_path = output_dir / "summary.json"
    summary_data = {
        "config": {
            "doc_types": doc_types,
            "count_per_type": args.count,
            "error_levels": error_levels,
            "degradation": degradation_params,
            "model": args.model,
            "seed": args.seed,
        },
        "documents": all_summaries,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Done! Output written to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Total documents: {len(all_summaries)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
