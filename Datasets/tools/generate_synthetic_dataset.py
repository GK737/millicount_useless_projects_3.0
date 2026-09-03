"""
Procedural Synthetic Millipede Dataset Generator for YOLO.
Generates realistic millipedes with controllable segment/ring counts,
articulated curves, natural chitin shading, legs, and varied backgrounds.
Outputs exact ground-truth labels in YOLO-OBB, YOLO-BBox, and YOLO-Seg formats.
"""

import os
import math
import random
import argparse
import numpy as np
import cv2
from pathlib import Path


def generate_background(width, height):
    """Generates realistic natural backgrounds (soil, stone, bark, litter)."""
    bg_type = random.choice(["soil", "stone", "bark", "leaf_litter"])

    if bg_type == "soil":
        base_color = np.array([random.randint(25, 45), random.randint(35, 60), random.randint(50, 80)], dtype=np.float32)
        noise = np.random.normal(0, 18, (height, width, 3)).astype(np.float32)
        bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        bg = cv2.GaussianBlur(bg, (5, 5), 0)
        # Add pebble specks
        for _ in range(random.randint(40, 100)):
            px = random.randint(0, width - 1)
            py = random.randint(0, height - 1)
            pr = random.randint(1, 4)
            pc = random.randint(100, 180)
            cv2.circle(bg, (px, py), pr, (pc, pc, pc), -1)

    elif bg_type == "stone":
        base_val = random.randint(110, 150)
        base_color = np.array([base_val - 5, base_val, base_val + 5], dtype=np.float32)
        noise = np.random.normal(0, 25, (height, width, 3)).astype(np.float32)
        bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        bg = cv2.GaussianBlur(bg, (7, 7), 0)
        # Add stone paver lines
        if random.random() < 0.6:
            pt1 = (0, random.randint(0, height))
            pt2 = (width, random.randint(0, height))
            cv2.line(bg, pt1, pt2, (random.randint(60, 90), random.randint(60, 90), random.randint(60, 90)), 3)

    elif bg_type == "bark":
        base_color = np.array([30, 45, 65], dtype=np.float32)
        noise = np.random.normal(0, 20, (height, width, 3)).astype(np.float32)
        bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        # Add fibrous longitudinal bark streaks
        streak_kernel = np.ones((1, random.choice([15, 25])), np.float32) / 20.0
        bg = cv2.filter2D(bg, -1, streak_kernel)

    else:  # leaf litter
        base_color = np.array([30, 50, 45], dtype=np.float32)
        noise = np.random.normal(0, 22, (height, width, 3)).astype(np.float32)
        bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        for _ in range(random.randint(15, 30)):
            lx = random.randint(0, width)
            ly = random.randint(0, height)
            lr = random.randint(15, 45)
            leaf_col = (random.randint(20, 50), random.randint(45, 90), random.randint(60, 110))
            cv2.ellipse(bg, (lx, ly), (lr, lr // 2), random.randint(0, 180), 0, 360, leaf_col, -1)
        bg = cv2.GaussianBlur(bg, (9, 9), 0)

    return bg


def generate_spline_points(num_segments, width, height):
    """Generates a smooth natural curved path for the millipede body."""
    curve_style = random.choice(["arc", "s_curve", "c_shape", "diagonal", "meander"])

    margin = 80
    if curve_style == "arc":
        p0 = np.array([random.uniform(margin, width * 0.4), random.uniform(margin, height * 0.4)])
        p1 = np.array([random.uniform(width * 0.3, width * 0.7), random.uniform(margin, height * 0.8)])
        p2 = np.array([random.uniform(width * 0.6, width - margin), random.uniform(height * 0.5, height - margin)])
        control_points = [p0, p1, p2]
    elif curve_style == "s_curve":
        p0 = np.array([random.uniform(margin, width * 0.3), random.uniform(margin, height * 0.4)])
        p1 = np.array([random.uniform(width * 0.3, width * 0.5), random.uniform(height * 0.6, height - margin)])
        p2 = np.array([random.uniform(width * 0.5, width * 0.7), random.uniform(margin, height * 0.4)])
        p3 = np.array([random.uniform(width * 0.7, width - margin), random.uniform(height * 0.5, height - margin)])
        control_points = [p0, p1, p2, p3]
    elif curve_style == "c_shape":
        cx, cy = random.uniform(width * 0.4, width * 0.6), random.uniform(height * 0.4, height * 0.6)
        radius = random.uniform(min(width, height) * 0.25, min(width, height) * 0.38)
        start_angle = random.uniform(0, math.pi * 2)
        total_angle = random.uniform(math.pi * 0.8, math.pi * 1.5)
        num_cp = 6
        control_points = []
        for i in range(num_cp):
            ang = start_angle + (total_angle * (i / (num_cp - 1)))
            control_points.append(np.array([cx + radius * math.cos(ang), cy + radius * math.sin(ang)]))
    else:  # diagonal or meander
        p0 = np.array([random.uniform(margin, width * 0.3), random.uniform(margin, height * 0.3)])
        p1 = np.array([random.uniform(width * 0.3, width * 0.6), random.uniform(height * 0.3, height * 0.7)])
        p2 = np.array([random.uniform(width * 0.6, width - margin), random.uniform(height * 0.6, height - margin)])
        control_points = [p0, p1, p2]

    # Evaluate dense spline
    t_samples = np.linspace(0, 1, 1000)
    cp = np.array(control_points)

    # Catmull-Rom or Bezier interpolation
    if len(control_points) == 3:
        # Quadratic Bezier
        dense_curve = np.array([(1-t)**2 * cp[0] + 2*(1-t)*t * cp[1] + t**2 * cp[2] for t in t_samples])
    else:
        # Cubic or higher Bezier
        n = len(control_points) - 1
        dense_curve = np.zeros((len(t_samples), 2))
        for i, pt in enumerate(control_points):
            binomial = math.comb(n, i)
            coeff = binomial * ((1 - t_samples)**(n - i)) * (t_samples**i)
            dense_curve += np.outer(coeff, pt)

    # Arc-length parameterization for uniform segment spacing
    diffs = np.diff(dense_curve, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_length = cum_dist[-1]

    # Sample num_segments points uniformly along arc length
    segment_spacing = total_length / (num_segments + 1)
    target_dists = np.linspace(segment_spacing * 0.5, total_length - segment_spacing * 0.5, num_segments)

    centers = []
    tangents = []
    normals = []

    for d in target_dists:
        idx = np.searchsorted(cum_dist, d)
        idx = min(max(idx, 1), len(dense_curve) - 2)
        pos = dense_curve[idx]
        tangent = dense_curve[idx + 1] - dense_curve[idx - 1]
        t_norm = np.linalg.norm(tangent)
        if t_norm > 1e-6:
            tangent /= t_norm
        else:
            tangent = np.array([1.0, 0.0])
        normal = np.array([-tangent[1], tangent[0]])

        centers.append(pos)
        tangents.append(tangent)
        normals.append(normal)

    return centers, tangents, normals, segment_spacing


def render_millipede(image, centers, tangents, normals, segment_spacing):
    """
    Renders millipede segments, legs, head, and telson with 3D shading.
    Returns:
      rendered_image: np.ndarray
      annotations: list of dicts with segment bbox, obb, polygon, class_id
    """
    img = image.copy()
    h, w = img.shape[:2]
    num_segments = len(centers)

    # Color scheme
    palette_name = random.choice(["dark_brown", "banded_yellow", "black_red", "orange_ring"])
    if palette_name == "dark_brown":
        body_base = np.array([15, 25, 45])       # BGR deep warm brown
        margin_col = np.array([10, 18, 30])      # darker suture
        dorsal_hl = np.array([45, 65, 95])       # specular sheen
        leg_col = (20, 40, 70)
    elif palette_name == "banded_yellow":
        body_base = np.array([15, 20, 25])       # dark body
        margin_col = np.array([30, 120, 180])    # bright yellow/gold ring edge
        dorsal_hl = np.array([60, 70, 80])
        leg_col = (40, 140, 190)
    elif palette_name == "black_red":
        body_base = np.array([18, 18, 22])       # obsidian black
        margin_col = np.array([20, 30, 120])     # scarlet red trim
        dorsal_hl = np.array([50, 50, 60])
        leg_col = (25, 35, 140)
    else:  # orange_ring
        body_base = np.array([20, 35, 60])
        margin_col = np.array([25, 90, 175])     # vibrant orange banding
        dorsal_hl = np.array([60, 80, 120])
        leg_col = (30, 80, 160)

    # Base body caliber/radius
    body_width_base = random.uniform(22.0, 34.0)
    segment_length = max(segment_spacing * 0.95, 6.0)

    annotations = []

    # 1. Render Shadow
    shadow_offset = np.array([random.uniform(4, 10), random.uniform(6, 12)])
    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(num_segments):
        c = centers[i] + shadow_offset
        t = tangents[i]
        n = normals[i]
        # taper at ends
        taper = math.sin((i + 0.5) / num_segments * math.pi) ** 0.35
        cur_w = body_width_base * taper
        cur_l = segment_length

        c1 = c + (t * (cur_l / 2)) - (n * (cur_w / 2))
        c2 = c + (t * (cur_l / 2)) + (n * (cur_w / 2))
        c3 = c - (t * (cur_l / 2)) + (n * (cur_w / 2))
        c4 = c - (t * (cur_l / 2)) - (n * (cur_w / 2))
        pts = np.array([c1, c2, c3, c4], dtype=np.int32)
        cv2.fillConvexPoly(shadow_mask, pts, 255)

    shadow_blur = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    shadow_alpha = (shadow_blur.astype(np.float32) / 255.0) * 0.45
    for c in range(3):
        img[:, :, c] = (img[:, :, c].astype(np.float32) * (1.0 - shadow_alpha)).astype(np.uint8)

    # 2. Render Legs (diplosegments bear 2 pairs of legs per segment)
    leg_phase = random.uniform(0, math.pi * 2)
    for i in range(2, num_segments - 2):
        c = centers[i]
        n = normals[i]
        t = tangents[i]
        taper = math.sin((i + 0.5) / num_segments * math.pi) ** 0.35
        cur_w = body_width_base * taper

        # Metachronal wave leg oscillation
        phase = leg_phase + i * 0.5
        leg_ext = math.sin(phase) * 3.0

        for side in [-1, 1]:  # Left and right sides
            base_pt1 = c - (t * 2) + (n * side * (cur_w * 0.45))
            mid_pt1 = base_pt1 + (n * side * (12 + leg_ext)) + (t * 2)
            tip_pt1 = mid_pt1 + (n * side * 6) - (t * 3)

            base_pt2 = c + (t * 2) + (n * side * (cur_w * 0.45))
            mid_pt2 = base_pt2 + (n * side * (12 - leg_ext)) + (t * 1)
            tip_pt2 = mid_pt2 + (n * side * 6) - (t * 2)

            for b_pt, m_pt, t_pt in [(base_pt1, mid_pt1, tip_pt1), (base_pt2, mid_pt2, tip_pt2)]:
                pts = np.array([b_pt, m_pt, t_pt], dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=False, color=leg_col, thickness=2, lineType=cv2.LINE_AA)

    # 3. Render Each Segment with 3D Cylindrical Shading & Grooves
    for i in range(num_segments):
        c = centers[i]
        t = tangents[i]
        n = normals[i]

        # Tapering near head (index 0) and telson (index num_segments - 1)
        progress = (i + 0.5) / num_segments
        taper = math.sin(progress * math.pi) ** 0.35
        if i < 3:
            taper = 0.65 + (i * 0.12)
        elif i > num_segments - 4:
            taper = 0.60 + ((num_segments - 1 - i) * 0.13)

        cur_w = body_width_base * taper
        cur_l = segment_length

        # 4 OBB corners (ordered counter-clockwise: top-left, top-right, bot-right, bot-left)
        c1 = c + (t * (cur_l / 2)) - (n * (cur_w / 2))
        c2 = c + (t * (cur_l / 2)) + (n * (cur_w / 2))
        c3 = c - (t * (cur_l / 2)) + (n * (cur_w / 2))
        c4 = c - (t * (cur_l / 2)) - (n * (cur_w / 2))

        obb_pts = np.array([c1, c2, c3, c4], dtype=np.float32)
        int_pts = np.array([c1, c2, c3, c4], dtype=np.int32)

        # Draw segment body
        cv2.fillConvexPoly(img, int_pts, [int(x) for x in body_base], lineType=cv2.LINE_AA)

        # Transverse dorsal highlight strip (specular highlight along midline)
        hl1 = c + (t * (cur_l * 0.3)) - (n * (cur_w * 0.18))
        hl2 = c + (t * (cur_l * 0.3)) + (n * (cur_w * 0.18))
        hl3 = c - (t * (cur_l * 0.3)) + (n * (cur_w * 0.18))
        hl4 = c - (t * (cur_l * 0.3)) - (n * (cur_w * 0.18))
        cv2.fillConvexPoly(img, np.array([hl1, hl2, hl3, hl4], dtype=np.int32), [int(x) for x in dorsal_hl], lineType=cv2.LINE_AA)

        # Intersegmental groove/suture ring (annulus line at posterior margin)
        annulus_p1 = c - (t * (cur_l / 2)) - (n * (cur_w / 2))
        annulus_p2 = c - (t * (cur_l / 2)) + (n * (cur_w / 2))
        cv2.line(img, tuple(annulus_p1.astype(int)), tuple(annulus_p2.astype(int)), [int(x) for x in margin_col], 2, lineType=cv2.LINE_AA)

        # Side margin color accent (e.g. paranota dots/bands)
        if palette_name in ["banded_yellow", "orange_ring"]:
            dot_l = c - (n * (cur_w * 0.44))
            dot_r = c + (n * (cur_w * 0.44))
            cv2.circle(img, tuple(dot_l.astype(int)), 2, [int(x) for x in margin_col], -1, lineType=cv2.LINE_AA)
            cv2.circle(img, tuple(dot_r.astype(int)), 2, [int(x) for x in margin_col], -1, lineType=cv2.LINE_AA)

        # Class ID assignment:
        # 0 = regular body segment / ring
        # 1 = head capsule (i == 0)
        # 2 = telson (i == num_segments - 1)
        if i == 0:
            class_id = 1
        elif i == num_segments - 1:
            class_id = 2
        else:
            class_id = 0

        # Compute axis-aligned bbox
        min_x = max(0.0, float(np.min(obb_pts[:, 0])))
        max_x = min(float(w), float(np.max(obb_pts[:, 0])))
        min_y = max(0.0, float(np.min(obb_pts[:, 1])))
        max_y = min(float(h), float(np.max(obb_pts[:, 1])))

        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox_cx = min_x + bbox_w / 2.0
        bbox_cy = min_y + bbox_h / 2.0

        annotations.append({
            "segment_index": i,
            "class_id": class_id,
            "center": (float(c[0]), float(c[1])),
            "tangent": (float(t[0]), float(t[1])),
            "normal": (float(n[0]), float(n[1])),
            "obb_corners": obb_pts,  # 4x2 coordinates
            "bbox": (bbox_cx, bbox_cy, bbox_w, bbox_h),  # absolute
            "bbox_norm": (bbox_cx / w, bbox_cy / h, bbox_w / w, bbox_h / h),
            "polygon_norm": [(pt[0] / w, pt[1] / h) for pt in obb_pts]
        })

    # 4. Render Head Details (Antennae and rounded anterior capsule)
    head_c = centers[0]
    head_t = tangents[0]
    head_n = normals[0]
    head_w = body_width_base * 0.65

    # Rounded collum / head cap
    cap_pt = head_c + (head_t * 8)
    cv2.circle(img, tuple(cap_pt.astype(int)), int(head_w * 0.45), [int(x) for x in body_base], -1, lineType=cv2.LINE_AA)

    # Antennae
    ant1_p1 = cap_pt - (head_n * 6)
    ant1_p2 = ant1_p1 + (head_t * 14) - (head_n * 8)
    ant1_p3 = ant1_p2 + (head_t * 8) + (head_n * 2)

    ant2_p1 = cap_pt + (head_n * 6)
    ant2_p2 = ant2_p1 + (head_t * 14) + (head_n * 8)
    ant2_p3 = ant2_p2 + (head_t * 8) - (head_n * 2)

    cv2.polylines(img, [np.array([ant1_p1, ant1_p2, ant1_p3], dtype=np.int32)], False, (15, 20, 30), 2, cv2.LINE_AA)
    cv2.polylines(img, [np.array([ant2_p1, ant2_p2, ant2_p3], dtype=np.int32)], False, (15, 20, 30), 2, cv2.LINE_AA)

    # 5. Natural post-processing (subtle blur, sensor noise)
    if random.random() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3), 0.5)

    return img, annotations


def save_yolo_labels(annotations, label_path, img_width, img_height, label_format="obb"):
    """
    Saves annotations to YOLO text format:
    - obb: class_id x1 y1 x2 y2 x3 y4 x4 y4 (normalized 0..1)
    - bbox: class_id x_center y_center width height (normalized 0..1)
    - seg: class_id x1 y1 x2 y2 ... xn yn (normalized 0..1 polygon)
    """
    lines = []
    for ann in annotations:
        cid = ann["class_id"]

        if label_format == "obb":
            # 4 normalized corner points
            corners = ann["obb_corners"]
            coords = []
            for pt in corners:
                nx = max(0.0, min(1.0, pt[0] / img_width))
                ny = max(0.0, min(1.0, pt[1] / img_height))
                coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
            lines.append(f"{cid} " + " ".join(coords))

        elif label_format == "seg":
            coords = []
            for pt in ann["polygon_norm"]:
                nx = max(0.0, min(1.0, pt[0]))
                ny = max(0.0, min(1.0, pt[1]))
                coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
            lines.append(f"{cid} " + " ".join(coords))

        else:  # bbox
            cx, cy, bw, bh = ann["bbox_norm"]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_dataset(num_images=50, output_root=".", label_format="obb", val_ratio=0.2, img_size=(640, 640)):
    """Generates synthetic dataset and splits into train/val."""
    root = Path(output_root)
    train_img_dir = root / "train" / "images"
    train_lbl_dir = root / "train" / "labels"
    val_img_dir = root / "val" / "images"
    val_lbl_dir = root / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating {num_images} synthetic millipede images (format={label_format})...")

    manifest = []
    width, height = img_size

    for i in range(num_images):
        is_val = (random.random() < val_ratio)
        img_dir = val_img_dir if is_val else train_img_dir
        lbl_dir = val_lbl_dir if is_val else train_lbl_dir
        split = "val" if is_val else "train"

        # Realistic segment count: 25 to 65 rings
        num_segments = random.randint(25, 65)

        bg = generate_background(width, height)
        centers, tangents, normals, segment_spacing = generate_spline_points(num_segments, width, height)
        rendered_img, annotations = render_millipede(bg, centers, tangents, normals, segment_spacing)

        file_stem = f"synth_milli_{i:04d}_seg{num_segments}"
        img_file = img_dir / f"{file_stem}.jpg"
        lbl_file = lbl_dir / f"{file_stem}.txt"

        cv2.imwrite(str(img_file), rendered_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        save_yolo_labels(annotations, lbl_file, width, height, label_format=label_format)

        manifest.append({
            "filename": f"{file_stem}.jpg",
            "split": split,
            "true_segment_count": num_segments,
            "width": width,
            "height": height
        })

        if (i + 1) % 10 == 0 or (i + 1) == num_images:
            print(f"    -> Generated [{i + 1}/{num_images}] ({split}): {file_stem} ({num_segments} segments)")

    print(f"\n[+] Successfully generated {num_images} images into '{output_root}'.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic millipede dataset with exact segment annotations.")
    parser.add_argument("--count", type=int, default=50, help="Number of images to generate (default: 50)")
    parser.add_argument("--format", type=str, choices=["obb", "bbox", "seg"], default="obb", help="YOLO label format (obb, bbox, seg)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set split ratio (default: 0.2)")
    parser.add_argument("--output", type=str, default=".", help="Dataset root path")
    args = parser.parse_args()

    generate_dataset(num_images=args.count, output_root=args.output, label_format=args.format, val_ratio=args.val_ratio)
