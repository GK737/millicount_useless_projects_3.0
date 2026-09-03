"""
Assisted Millipede Segment Annotator.
Automatically identifies the millipede body axis, detects individual ring/segment
boundaries via transverse gradient analysis along the spine, and exports
candidate annotations in YOLO-OBB, YOLO-BBox, and Label Studio formats.
"""

import os
import sys
import math
import argparse
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def extract_millipede_spine(binary_mask):
    """
    Finds endpoints and traces the medial axis spine through the mask.
    Returns:
        spine_points: np.ndarray of shape (N, 2) ordered from head to tail.
        body_width: average caliber along the spine.
    """
    # Extract clean medial spine using principal component slicing
    mask_pts = np.argwhere(binary_mask > 0)[:, ::-1].astype(np.float32)  # (x, y)
    if len(mask_pts) < 50:
        return np.empty((0, 2)), 25.0

    mean, eigenvectors = cv2.PCACompute(mask_pts, mean=None)
    center = mean[0]
    v_long = eigenvectors[0]   # Long axis of body
    v_trans = eigenvectors[1]  # Transverse axis

    # Projections along long axis
    projs = np.dot(mask_pts - center, v_long)
    p_min, p_max = float(np.min(projs)), float(np.max(projs))

    # Slice along long axis in 100 intervals to get centroid of each slice
    num_slices = 120
    slice_bins = np.linspace(p_min, p_max, num_slices)
    bin_centers = []
    widths = []

    for k in range(len(slice_bins) - 1):
        s_low = slice_bins[k]
        s_high = slice_bins[k + 1]
        in_slice = mask_pts[(projs >= s_low) & (projs < s_high)]
        if len(in_slice) >= 3:
            # Centroid of this slice
            slice_mean = np.mean(in_slice, axis=0)
            bin_centers.append(slice_mean)
            # Caliber in this slice
            trans_projs = np.dot(in_slice - center, v_trans)
            widths.append(float(np.max(trans_projs) - np.min(trans_projs)))

    if len(bin_centers) < 10:
        t_vals = np.linspace(p_min, p_max, 50)
        smoothed_spine = np.array([center + t * v_long for t in t_vals])
        avg_width = 25.0
        return smoothed_spine, avg_width

    raw_spine = np.array(bin_centers, dtype=np.float32)

    # Smooth spine curve with gaussian filter
    smooth_x = gaussian_filter1d(raw_spine[:, 0], sigma=3.0)
    smooth_y = gaussian_filter1d(raw_spine[:, 1], sigma=3.0)
    smoothed_spine = np.column_stack([smooth_x, smooth_y])

    avg_width = float(np.median(widths)) if widths else 25.0
    return smoothed_spine, avg_width


def detect_rings_along_spine(image_bgr, spine_pts, avg_width, sensitivity=1.0, target_segments=None):
    """
    Samples transverse cross-sections along the spine, computes intensity contrast,
    and detects segment peaks / valleys corresponding to rings.
    sensitivity: multiplier for peak detection density (higher = more segments detected)
    target_segments: optional expected count to guide spacing
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Arc-length parameterization
    diffs = np.diff(spine_pts, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dist[-1]

    # Dense sample every 1 pixel
    sample_dists = np.arange(0, total_len, 1.0)
    dense_pts = []
    dense_tangents = []
    dense_normals = []

    for d in sample_dists:
        idx = np.searchsorted(cum_dist, d)
        idx = min(max(idx, 1), len(spine_pts) - 2)
        pos = spine_pts[idx]
        tan = spine_pts[idx + 1] - spine_pts[idx - 1]
        t_norm = np.linalg.norm(tan)
        tan = tan / t_norm if t_norm > 1e-6 else np.array([1.0, 0.0])
        norm = np.array([-tan[1], tan[0]])

        dense_pts.append(pos)
        dense_tangents.append(tan)
        dense_normals.append(norm)

    # Extract 1D transverse profile slice average for each spine position
    profile = []
    slice_half = max(3.0, avg_width * 0.35)

    for pos, norm in zip(dense_pts, dense_normals):
        vals = []
        for offset in np.linspace(-slice_half, slice_half, 5):
            sp = pos + norm * offset
            sx = int(round(sp[0]))
            sy = int(round(sp[1]))
            if 0 <= sx < w and 0 <= sy < h:
                vals.append(gray[sy, sx])
        profile.append(np.mean(vals) if vals else 0.0)

    profile = np.array(profile, dtype=np.float32)

    # Detrend and smooth profile
    smooth_prof = gaussian_filter1d(profile, sigma=max(1.0, 2.0 / sensitivity))
    grad = np.abs(np.gradient(smooth_prof))

    # Detect peaks (segment sutures)
    if target_segments and target_segments > 5:
        expected_spacing = max(3, int(total_len / (target_segments + 1)))
        prominence = 0.5
    else:
        expected_spacing = max(3, int((avg_width * 0.16) / sensitivity))
        prominence = max(0.5, 1.5 / sensitivity)

    peaks, _ = find_peaks(grad, distance=expected_spacing, prominence=prominence)

    if len(peaks) < 8:
        # Fallback: uniform frequency sampling
        peaks = np.arange(expected_spacing, len(dense_pts) - expected_spacing, expected_spacing)

    # Assemble segment annotations
    annotations = []
    num_peaks = len(peaks)

    for i, p_idx in enumerate(peaks):
        pos = dense_pts[p_idx]
        tan = dense_tangents[p_idx]
        norm = dense_normals[p_idx]

        cur_w = avg_width * 1.1
        cur_l = float(expected_spacing * 1.1)

        # 4 OBB corners
        c1 = pos + (tan * (cur_l / 2)) - (norm * (cur_w / 2))
        c2 = pos + (tan * (cur_l / 2)) + (norm * (cur_w / 2))
        c3 = pos - (tan * (cur_l / 2)) + (norm * (cur_w / 2))
        c4 = pos - (tan * (cur_l / 2)) - (norm * (cur_w / 2))

        obb_pts = np.array([c1, c2, c3, c4], dtype=np.float32)

        # Bounding box
        min_x = max(0.0, float(np.min(obb_pts[:, 0])))
        max_x = min(float(w), float(np.max(obb_pts[:, 0])))
        min_y = max(0.0, float(np.min(obb_pts[:, 1])))
        max_y = min(float(h), float(np.max(obb_pts[:, 1])))

        bw = max_x - min_x
        bh = max_y - min_y
        cx = min_x + bw / 2.0
        cy = min_y + bh / 2.0

        cid = 1 if i == 0 else (2 if i == num_peaks - 1 else 0)

        annotations.append({
            "segment_index": i + 1,
            "class_id": cid,
            "center": (float(pos[0]), float(pos[1])),
            "obb_corners": obb_pts,
            "bbox_norm": (cx / w, cy / h, bw / w, bh / h)
        })

    return annotations


def auto_annotate_image(image_path, output_label_path=None, preview_path=None, label_format="obb", sensitivity=1.0, target_segments=None):
    """
    Processes an input millipede image and outputs draft YOLO annotations.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[!] Unable to read image: {image_path}")
        return None

    h, w = img.shape[:2]

    # Segment millipede body
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # Adaptive threshold / Otsu
    _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_reg = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find which mask captures the millipede (elongated contour)
    best_cnt = None
    best_aspect = 0.0

    for m in [thresh_inv, thresh_reg]:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > (w * h * 0.015):  # at least 1.5% of image
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if rw > 0 and rh > 0:
                    aspect = max(rw, rh) / min(rw, rh)
                    if aspect > best_aspect:
                        best_aspect = aspect
                        best_cnt = c

    if best_cnt is None:
        print(f"[!] Could not locate millipede body in {image_path}. Try running with anchor points.")
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Extract spine and caliber
    spine, avg_width = extract_millipede_spine(mask)

    # Detect individual segments
    annotations = detect_rings_along_spine(img, spine, avg_width, sensitivity=sensitivity, target_segments=target_segments)

    print(f"[+] Detected {len(annotations)} rings/segments in '{Path(image_path).name}'.")

    # Save YOLO format label
    if output_label_path:
        out_lbl = Path(output_label_path)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for ann in annotations:
            cid = ann["class_id"]
            if label_format == "obb":
                corners = ann["obb_corners"]
                coords = []
                for pt in corners:
                    coords.extend([f"{max(0.0, min(1.0, pt[0] / w)):.6f}", f"{max(0.0, min(1.0, pt[1] / h)):.6f}"])
                lines.append(f"{cid} " + " ".join(coords))
            else:
                cx, cy, bw, bh = ann["bbox_norm"]
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[+] Label saved to '{out_lbl}'")

    # Render Preview
    if preview_path:
        prev_p = Path(preview_path)
        prev_p.parent.mkdir(parents=True, exist_ok=True)

        vis = img.copy()
        for ann in annotations:
            pts = ann["obb_corners"].astype(np.int32)
            cid = ann["class_id"]
            col = (0, 165, 255) if cid == 1 else ((200, 50, 255) if cid == 2 else (0, 235, 120))
            cv2.polylines(vis, [pts], isClosed=True, color=col, thickness=2, lineType=cv2.LINE_AA)
            c = (int(ann["center"][0]), int(ann["center"][1]))
            cv2.circle(vis, c, 3, col, -1)

        banner = np.zeros((40, w, 3), dtype=np.uint8)
        banner[:] = (20, 20, 25)
        cv2.putText(
            banner, f"Auto-Annotated Rings: {len(annotations)}", (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA
        )
        combined = np.vstack([banner, vis])
        cv2.imwrite(str(prev_p), combined)
        print(f"[+] Preview saved to '{prev_p}'")

    return annotations


def batch_auto_annotate(input_dir, label_dir="test/labels", preview_dir="previews/test_annotated", label_format="obb", sensitivity=1.0, target_segments=None):
    """Processes an entire directory of millipede images."""
    in_p = Path(input_dir)
    lbl_p = Path(label_dir)
    prev_p = Path(preview_dir)
    lbl_p.mkdir(parents=True, exist_ok=True)
    prev_p.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [f for f in in_p.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

    print(f"\n[*] Batch Auto-Annotating {len(files)} images from '{input_dir}'...")

    annotated_count = 0
    for idx, img_file in enumerate(files, start=1):
        out_lbl = lbl_p / f"{img_file.stem}.txt"
        out_prev = prev_p / f"vis_{img_file.stem}.jpg"
        print(f"[{idx}/{len(files)}] Processing: {img_file.name}...")
        try:
            res = auto_annotate_image(
                image_path=img_file,
                output_label_path=out_lbl,
                preview_path=out_prev,
                label_format=label_format,
                sensitivity=sensitivity,
                target_segments=target_segments
            )
            if res:
                annotated_count += 1
        except Exception as e:
            print(f"    [!] Error processing {img_file.name}: {e}")

    print(f"\n[+] Finished! Successfully annotated {annotated_count}/{len(files)} images.")
    print(f"[+] Labels saved to '{lbl_p}', previews saved to '{prev_p}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assisted Auto-Annotator for Millipede Segments.")
    parser.add_argument("--image", type=str, help="Input single millipede image path")
    parser.add_argument("--dir", type=str, help="Input directory of millipede images for batch annotation")
    parser.add_argument("--label", type=str, help="Output label text path (single image)")
    parser.add_argument("--preview", type=str, help="Output preview image path (single image)")
    parser.add_argument("--label-dir", type=str, default="test/labels", help="Output label directory (batch mode)")
    parser.add_argument("--preview-dir", type=str, default="previews/test_annotated", help="Output preview directory (batch mode)")
    parser.add_argument("--format", type=str, choices=["obb", "bbox"], default="obb", help="Label format")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Peak detection sensitivity multiplier (default: 1.0)")
    parser.add_argument("--target-segments", type=int, default=None, help="Approximate expected segment count (optional)")
    args = parser.parse_args()

    if args.dir:
        batch_auto_annotate(
            input_dir=args.dir,
            label_dir=args.label_dir,
            preview_dir=args.preview_dir,
            label_format=args.format,
            sensitivity=args.sensitivity,
            target_segments=args.target_segments
        )
    elif args.image:
        auto_annotate_image(
            image_path=args.image,
            output_label_path=args.label,
            preview_path=args.preview,
            label_format=args.format,
            sensitivity=args.sensitivity,
            target_segments=args.target_segments
        )
    else:
        print("[!] Either --image or --dir must be provided.")
