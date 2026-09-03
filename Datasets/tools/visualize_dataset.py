"""
Visualize and Verify Millipede Dataset Annotations for YOLO.
Renders bounding boxes / oriented bounding boxes / polygons,
overlays sequential ring numbers along the millipede, and confirms segment counts.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2


CLASS_NAMES = {
    0: "segment",
    1: "head",
    2: "telson"
}

CLASS_COLORS = {
    0: (0, 230, 115),   # Bright Green / Teal for body segments
    1: (0, 165, 255),   # Bright Orange for Head
    2: (200, 50, 255)   # Magenta for Telson
}


def parse_yolo_label_line(line, img_w, img_h):
    """
    Parses a single label line into (class_id, points_xy, box_type, center).
    Supports:
    - OBB: 9 tokens (class + 8 coords)
    - BBox: 5 tokens (class + cx cy w h)
    - Polygon: 2N + 1 tokens
    """
    parts = line.strip().split()
    if not parts:
        return None

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    if len(coords) == 8:
        # YOLO-OBB: 4 corner points
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        center = np.mean(pts, axis=0)
        return {
            "class_id": class_id,
            "type": "obb",
            "pts": pts.astype(np.int32),
            "center": center
        }

    elif len(coords) == 4:
        # Standard YOLO BBox: cx, cy, w, h
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
        return {
            "class_id": class_id,
            "type": "bbox",
            "pts": pts,
            "center": np.array([abs_cx, abs_cy])
        }

    elif len(coords) >= 6 and len(coords) % 2 == 0:
        # YOLO Segmentation Polygon
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        center = np.mean(pts, axis=0)
        return {
            "class_id": class_id,
            "type": "poly",
            "pts": pts.astype(np.int32),
            "center": center
        }

    return None


def visualize_sample(image_path, label_path, output_path=None):
    """Renders visual verification of an annotated millipede sample."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[!] Could not load image: {image_path}")
        return

    h, w = img.shape[:2]

    if not Path(label_path).exists():
        print(f"[!] Label file missing: {label_path}")
        return

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    items = []
    for line in lines:
        parsed = parse_yolo_label_line(line, w, h)
        if parsed:
            items.append(parsed)

    # Sort items sequentially:
    # If head (class 1) exists, find head center and traverse along closest neighbors
    sorted_items = sort_segments_spatially(items)

    total_segments = len(sorted_items)
    head_count = sum(1 for it in items if it["class_id"] == 1)
    body_count = sum(1 for it in items if it["class_id"] == 0)
    telson_count = sum(1 for it in items if it["class_id"] == 2)

    vis_img = img.copy()

    # Draw spine connecting centers
    if len(sorted_items) > 1:
        spine_pts = np.array([it["center"].astype(np.int32) for it in sorted_items])
        cv2.polylines(vis_img, [spine_pts], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Draw segment boxes and numbers
    for idx, it in enumerate(sorted_items, start=1):
        cid = it["class_id"]
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        pts = it["pts"]

        # Draw polygon/box
        cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Draw small index circle and number
        c = it["center"].astype(int)
        cv2.circle(vis_img, tuple(c), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis_img, tuple(c), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # Label tag every few segments or for head/tail to prevent visual clutter
        if idx == 1 or idx == total_segments or idx % 5 == 0:
            text = f"{idx}"
            cv2.putText(
                vis_img, text, (c[0] + 8, c[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                vis_img, text, (c[0] + 8, c[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
            )

    # Draw Header Stats Banner
    banner_h = 48
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (20, 20, 25)

    title_text = f"Total Rings Counted: {total_segments}"
    details_text = f"Head: {head_count} | Body Segments: {body_count} | Telson: {telson_count}"

    cv2.putText(banner, title_text, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(banner, details_text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    combined = np.vstack([banner, vis_img])

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), combined)
        print(f"[+] Saved visualization to '{out_p}'")

    return combined, total_segments


def sort_segments_spatially(items):
    """
    Sorts detected items in anatomical order from head to telson.
    Uses nearest-neighbor trajectory following along the body spine.
    """
    if not items:
        return []

    # Find head (class_id == 1) if available
    head_candidates = [it for it in items if it["class_id"] == 1]
    if head_candidates:
        start_node = head_candidates[0]
    else:
        # Otherwise pick the most extreme point on the convex hull
        centers = np.array([it["center"] for it in items])
        # Pick point with minimum x or y as start
        start_idx = np.argmin(centers[:, 0] + centers[:, 1])
        start_node = items[start_idx]

    unvisited = [it for it in items if it is not start_node]
    ordered = [start_node]
    current = start_node

    while unvisited:
        cur_c = current["center"]
        # Find closest unvisited center
        dists = [np.linalg.norm(cur_c - u["center"]) for u in unvisited]
        min_idx = int(np.argmin(dists))
        next_node = unvisited.pop(min_idx)
        ordered.append(next_node)
        current = next_node

    return ordered


def inspect_dataset(dataset_root=".", split="train", max_samples=5, output_dir="previews"):
    """Inspects multiple samples in a split and saves visual previews."""
    img_dir = Path(dataset_root) / split / "images"
    lbl_dir = Path(dataset_root) / split / "labels"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"[!] Directory not found: {img_dir}")
        return

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print(f"[!] No images found in {img_dir}")
        return

    print(f"[*] Visualizing {min(len(images), max_samples)} samples from '{split}'...")

    for img_path in images[:max_samples]:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        out_path = out_dir / f"vis_{img_path.stem}.jpg"
        visualize_sample(img_path, lbl_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Millipede Dataset and count rings.")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--label", type=str, help="Path to single label file")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to inspect (train, val, test)")
    parser.add_argument("--root", type=str, default=".", help="Dataset root path")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--output", type=str, default="previews", help="Preview output folder")
    args = parser.parse_args()

    if args.image and args.label:
        out_name = Path(args.output) / f"vis_{Path(args.image).stem}.jpg"
        visualize_sample(args.image, args.label, out_name)
    else:
        inspect_dataset(dataset_root=args.root, split=args.split, max_samples=args.samples, output_dir=args.output)
