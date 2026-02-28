from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
from PIL import Image

from .core.engine import OMRScanner, OMRConfig, load_scoring_key_from_bytes
from .core.security import ensure_within_dir


def _write_outputs(out_dir: Path, scan_result, scoring_key_path: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # responses.csv
    csv_path = ensure_within_dir(out_dir / "responses.csv", out_dir)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("item,choice\n")
        for item in range(1, 241):
            v = scan_result.responses_final.get(item, -1)
            f.write(f"{item},{v}\n")

    # scores.json
    scores = {
        "scan_id": scan_result.scan_id,
        "protocol": scan_result.protocol,
        "facette_scores": scan_result.facette_scores,
        "domain_scores": scan_result.domain_scores,
    }
    scores_path = ensure_within_dir(out_dir / "scores.json", out_dir)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # audit.json
    audit = {
        "scan_id": scan_result.scan_id,
        "diagnostics": scan_result.diagnostics,
        "meta": scan_result.meta,  # includes confidence, blank/ambiguous
    }
    audit_path = ensure_within_dir(out_dir / "audit.json", out_dir)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # overlay.png
    overlay_path = ensure_within_dir(out_dir / "overlay.png", out_dir)
    cv2.imwrite(str(overlay_path), scan_result.overlay_bgr)

    # debug_mask.png
    dbg_path = ensure_within_dir(out_dir / "debug_mask.png", out_dir)
    cv2.imwrite(str(dbg_path), scan_result.debug_mask)

    # warped.png
    warped_path = ensure_within_dir(out_dir / "warped.png", out_dir)
    cv2.imwrite(str(warped_path), scan_result.warped_bgr)


def _load_scoring_key(path: Path) -> Dict[int, List[int]]:
    data = path.read_bytes()
    return load_scoring_key_from_bytes(data)


def cmd_scan(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    out_dir = Path(args.out)
    scoring_key_path = Path(args.scoring_key)

    scoring_key = _load_scoring_key(scoring_key_path)

    cfg = OMRConfig(
        mark_threshold=args.mark_threshold,
        ambiguity_gap=args.ambiguity_gap,
        detect_blue=not args.no_blue,
        detect_black=not args.no_black,
        black_dark_thresh=args.black_dark_thresh,
        black_baseline_quantile=args.black_baseline_quantile,
    )

    scanner = OMRScanner(cfg=cfg)

    pil_img = Image.open(image_path)
    res = scanner.scan_pil(pil_img, scoring_key)

    _write_outputs(out_dir, res)
    print(f"OK: {out_dir} (scan_id={res.scan_id})")

    if args.fail_on_invalid and not res.protocol.get("valid", True):
        print("Protocol invalid")
        return 2
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    in_dir = Path(args.input)
    out_dir = Path(args.out)
    scoring_key_path = Path(args.scoring_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    scoring_key = _load_scoring_key(scoring_key_path)

    cfg = OMRConfig(
        mark_threshold=args.mark_threshold,
        ambiguity_gap=args.ambiguity_gap,
        detect_blue=not args.no_blue,
        detect_black=not args.no_black,
        black_dark_thresh=args.black_dark_thresh,
        black_baseline_quantile=args.black_baseline_quantile,
    )

    scanner = OMRScanner(cfg=cfg)

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        images.extend(sorted(in_dir.glob(ext)))

    if not images:
        print("No images found")
        return 1

    summary = []
    for img_path in images:
        try:
            pil_img = Image.open(img_path)
            res = scanner.scan_pil(pil_img, scoring_key)
            one_out = out_dir / img_path.stem
            _write_outputs(one_out, res)
            summary.append({
                "file": str(img_path.name),
                "scan_id": res.scan_id,
                "valid": bool(res.protocol.get("valid", True)),
                "n_blank": int(res.protocol.get("n_blank", 0)),
                "ambiguous": int(res.diagnostics["stats"]["ambiguous"]),
            })
            print(f"OK: {img_path.name} -> {one_out}")
        except Exception as e:
            summary.append({"file": str(img_path.name), "error": str(e)})
            print(f"ERR: {img_path.name}: {e}")

    with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neo-pir-omr", description="NEO PI-R OMR Scanner")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser):
        sp.add_argument("--scoring-key", default=str(Path(__file__).resolve().parent / "data" / "scoring_key.csv"), help="Path to scoring_key.csv")
        sp.add_argument("--mark-threshold", type=float, default=1.7)
        sp.add_argument("--ambiguity-gap", type=float, default=0.9)
        sp.add_argument("--black-dark-thresh", type=int, default=110)
        sp.add_argument("--black-baseline-quantile", type=float, default=15.0)
        sp.add_argument("--no-blue", action="store_true")
        sp.add_argument("--no-black", action="store_true")

    s1 = sub.add_parser("scan", help="Scan a single image")
    s1.add_argument("--image", required=True)
    s1.add_argument("--out", required=True)
    s1.add_argument("--fail-on-invalid", action="store_true")
    add_common(s1)
    s1.set_defaults(func=cmd_scan)

    s2 = sub.add_parser("batch", help="Scan all images in a folder")
    s2.add_argument("--input", required=True)
    s2.add_argument("--out", required=True)
    add_common(s2)
    s2.set_defaults(func=cmd_batch)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
