"""
Millipede Segment & Ring Counter using YOLO.
Inference and counting engine:
1. Runs YOLOv8/v11 (OBB or BBox) detection.
2. Orders detected segments along the body spine from head to telson.
3. Detects inter-segment distance anomalies (occlusion/missed segment gaps).
4. Outputs final segment count and an annotated report image.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


CLASS_COLORS = {
    0: (0, 235, 120),   # Body segment
    1: (0, 165, 255),   # Head
    2: (210, 50, 255)   # Telson
}


def sort_segments_along_spine(detections):
    """
    Sorts detected items in anatomical order from head to telson.
    """
    if not detections:
        return []

    # 1. Identify head if present
    head_candidates = [d for d in detections if d["class_id"] == 1]
    if head_candidates:
        head_candidates.sort(key=lambda x: x["conf"], reverse=True)
        start_node = head_candidates[0]
    else:
        # Fallback: pick the end of the principal axis
        centers = np.array([d["center"] for d in detections])
        start_idx = np.argmin(centers[:, 0] + centers[:, 1])
        start_node = detections[start_idx]

    unvisited = [d for d in detections if d is not start_node]
    ordered = [start_node]
    current = start_node

    while unvisited:
        cur_c = current["center"]
        dists = [np.linalg.norm(cur_c - u["center"]) for u in unvisited]
        min_idx = int(np.argmin(dists))
        next_node = unvisited.pop(min_idx)
        ordered.append(next_node)
        current = next_node

    return ordered


def analyze_spacing_and_gaps(ordered_detections):
    """
    Analyzes inter-segment Euclidean distances to detect potential missed rings.
    """
    if len(ordered_detections) < 3:
        return []

    centers = np.array([d["center"] for d in ordered_detections])
    diffs = np.diff(centers, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    median_dist = float(np.median(dists))

    gaps = []
    for i, d in enumerate(dists):
        # If distance between adjacent segments is > 1.8x median spacing, likely a missed segment
        if d > (median_dist * 1.85):
            estimated_missing = int(round((d - median_dist) / median_dist))
            gaps.append({
                "between_indices": (i + 1, i + 2),
                "distance": d,
                "median_distance": median_dist,
                "estimated_missing": max(1, estimated_missing)
            })

    return gaps


def parse_label_file(label_path, img_w, img_h):
    """Parses ground-truth YOLO label file into detections format."""
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    detections = []
    for line in lines:
        parts = line.split()
        cid = int(parts[0])
        coords = [float(x) for x in parts[1:]]

        if len(coords) == 8:
            # OBB
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            center = np.mean(pts, axis=0)
            detections.append({
                "class_id": cid,
                "conf": 1.0,
                "pts": pts.astype(np.int32),
                "center": center
            })
        elif len(coords) == 4:
            # BBox
            cx, cy, bw, bh = coords
            abs_cx = cx * img_w
            abs_cy = cy * img_h
            abs_w = bw * img_w
            abs_h = bh * img_h
            x1 = int(abs_cx - abs_w / 2)
            y1 = int(abs_cy - abs_h / 2)
            x2 = int(abs_cx + abs_w / 2)
            y2 = int(abs_cy + abs_h / 2)
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            detections.append({
                "class_id": cid,
                "conf": 1.0,
                "pts": pts,
                "center": np.array([abs_cx, abs_cy])
            })
    return detections


def run_counting(image_path, model_path=None, label_path=None, conf_thresh=0.25, output_image_path=None):
    """
    Main counting pipeline.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[!] Could not load image: {image_path}")
        return None

    h, w = img.shape[:2]
    detections = []

    if model_path and Path(model_path).exists():
        if YOLO is None:
            print("[!] Ultralytics YOLO is not installed.")
            return None
        print(f"[*] Loading model weights from '{model_path}'...")
        model = YOLO(model_path)
        results = model.predict(img, conf=conf_thresh, verbose=False)[0]

        # Extract OBB or Boxes
        if hasattr(results, "obb") and results.obb is not None and len(results.obb) > 0:
            for obb in results.obb:
                cid = int(obb.cls[0])
                cf = float(obb.conf[0])
                poly = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)
                center = np.mean(poly, axis=0)
                detections.append({
                    "class_id": cid,
                    "conf": cf,
                    "pts": poly,
                    "center": center
                })
        elif hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cid = int(box.cls[0])
                cf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                detections.append({
                    "class_id": cid,
                    "conf": cf,
                    "pts": pts,
                    "center": center
                })

    elif label_path and Path(label_path).exists():
        print(f"[*] Reading ground-truth labels from '{label_path}'...")
        detections = parse_label_file(label_path, w, h)
    else:
        print("[!] Either --model or --label must be provided.")
        return None

    # Sort detections along body spine
    ordered = sort_segments_along_spine(detections)
    total_count = len(ordered)
    gaps = analyze_spacing_and_gaps(ordered)
    missing_gap_sum = sum(g["estimated_missing"] for g in gaps)
    adjusted_count = total_count + missing_gap_sum

    head_count = sum(1 for d in ordered if d["class_id"] == 1)
    body_count = sum(1 for d in ordered if d["class_id"] == 0)
    telson_count = sum(1 for d in ordered if d["class_id"] == 2)

    print(f"\n================ MILLIPEDE SEGMENT COUNT REPORT ================")
    print(f" Detected Total Rings: {total_count}")
    print(f" Breakdown: Head: {head_count} | Body Rings: {body_count} | Telson: {telson_count}")
    if gaps:
        print(f" [!] Warning: {len(gaps)} potential gap(s) detected along body curve:")
        for g in gaps:
            b1, b2 = g["between_indices"]
            print(f"     - Gap between segment #{b1} and #{b2}: ~{g['estimated_missing']} missing segment(s)")
        print(f" Estimated True Count (with gaps): {adjusted_count}")
    else:
        print(" [+] Segment spacing is continuous. No missing gaps detected.")
    print(f"=================================================================\n")

    # Render Visual Overlay
    vis = img.copy()

    # Draw spine
    if len(ordered) > 1:
        spine_pts = np.array([d["center"].astype(np.int32) for d in ordered])
        cv2.polylines(vis, [spine_pts], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Draw segments
    for idx, d in enumerate(ordered, start=1):
        cid = d["class_id"]
        col = CLASS_COLORS.get(cid, (255, 255, 255))
        pts = d["pts"]

        cv2.polylines(vis, [pts], isClosed=True, color=col, thickness=2, lineType=cv2.LINE_AA)
        c = tuple(d["center"].astype(int))
        cv2.circle(vis, c, 5, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, c, 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        if idx == 1 or idx == total_count or idx % 5 == 0:
            tag = f"{idx}"
            cv2.putText(vis, tag, (c[0] + 6, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, tag, (c[0] + 6, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Top Dashboard Banner
    banner = np.zeros((52, w, 3), dtype=np.uint8)
    banner[:] = (18, 18, 22)
    cv2.putText(
        banner, f"Total Detected Rings: {total_count}" + (f" (Est. {adjusted_count} with gaps)" if gaps else ""),
        (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 180), 2, cv2.LINE_AA
    )
    cv2.putText(
        banner, f"Head: {head_count} | Rings: {body_count} | Telson: {telson_count} | Gaps: {len(gaps)}",
        (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA
    )

    combined = np.vstack([banner, vis])

    if output_image_path:
        out_p = Path(output_image_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), combined)
        print(f"[+] Annotated report saved to '{out_p}'")

    return {
        "total_count": total_count,
        "adjusted_count": adjusted_count,
        "head_count": head_count,
        "body_count": body_count,
        "telson_count": telson_count,
        "gaps": gaps
    }


def run_batch_counting(input_dir, model_path=None, conf_thresh=0.25, output_dir="previews/test_results"):
    """Runs segment counting across an entire directory of test images."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = [f for f in in_path.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

    if not image_files:
        print(f"[!] No valid image files found in '{input_dir}'")
        return []

    print(f"\n================ BATCH MILLIPEDE SEGMENT TESTING ================")
    print(f" Target Directory : {input_dir}")
    print(f" Total Images     : {len(image_files)}")
    print(f" Output Directory : {output_dir}")
    print(f"=================================================================\n")

    summary_records = []

    for idx, img_p in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_p.name}...")
        report_path = out_path / f"report_{img_p.stem}.jpg"
        lbl_p = in_path.parent / "labels" / f"{img_p.stem}.txt"
        label_arg = str(lbl_p) if lbl_p.exists() else None

        res = run_counting(
            image_path=img_p,
            model_path=model_path,
            label_path=label_arg,
            conf_thresh=conf_thresh,
            output_image_path=report_path
        )

        if res:
            summary_records.append({
                "filename": img_p.name,
                "total_count": res["total_count"],
                "adjusted_count": res["adjusted_count"],
                "head": res["head_count"] > 0,
                "telson": res["telson_count"] > 0,
                "gaps": len(res["gaps"])
            })

    # Print summary table
    print(f"\n{'Filename':<45} | {'Detected':<8} | {'Est. w/ Gaps':<12} | {'Head':<5} | {'Telson':<6} | {'Gaps':<4}")
    print("-" * 90)
    for r in summary_records:
        print(f"{r['filename'][:44]:<45} | {r['total_count']:<8} | {r['adjusted_count']:<12} | {str(r['head']):<5} | {str(r['telson']):<6} | {r['gaps']:<4}")

    return summary_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count millipede rings/segments from YOLO detections.")
    parser.add_argument("--image", type=str, help="Input single millipede image path")
    parser.add_argument("--dir", type=str, help="Input directory of test images")
    parser.add_argument("--model", type=str, help="Path to trained YOLO model weights (.pt)")
    parser.add_argument("--label", type=str, help="Path to ground truth YOLO label (.txt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO (default: 0.25)")
    parser.add_argument("--output", type=str, default="previews/count_report.jpg", help="Output annotated report image path")
    parser.add_argument("--output-dir", type=str, default="previews/test_results", help="Output directory for batch mode")
    args = parser.parse_args()

    if args.dir:
        run_batch_counting(
            input_dir=args.dir,
            model_path=args.model,
            conf_thresh=args.conf,
            output_dir=args.output_dir
        )
    elif args.image:
        run_counting(
            image_path=args.image,
            model_path=args.model,
            label_path=args.label,
            conf_thresh=args.conf,
            output_image_path=args.output
        )
    else:
        print("[!] Either --image or --dir must be provided.")
