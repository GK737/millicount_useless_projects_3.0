"""
Core Millipede Computer Vision & Counting Engine.

Provides:
1. MillipedeOBBDetector: Deep Learning Oriented Bounding Box detector trained on millipede anatomy.
2. SpineOrderer: Anatomical head-to-telson sequential tracking along the curved body axis.
3. GapAnalyzer: Inter-segment distance anomaly & occlusion detector for true count estimation.
4. DashboardRenderer: High-resolution annotated overlays with class coloring and ring badges.
5. ClassicalProfiler: Signal-processing & unrolled body analyzer for baseline comparison.
"""

import os
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


CLASS_METADATA = {
    0: {"name": "Body Ring", "color_bgr": (120, 235, 0), "color_rgb": (0, 235, 120)},    # Emerald green
    1: {"name": "Head",      "color_bgr": (0, 165, 255), "color_rgb": (255, 165, 0)},    # Amber orange
    2: {"name": "Telson",    "color_bgr": (255, 50, 210), "color_rgb": (210, 50, 255)}   # Magenta
}


class ImagePreprocessor:
    """Preprocesses input frames with CLAHE, bilateral/gaussian filtering, and sharpening."""
    def __init__(self, blur_kernel=3, contrast_clip=2.0, sharpen=True):
        self.blur_kernel = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
        self.contrast_clip = contrast_clip
        self.sharpen = sharpen

    def process(self, image_bgr):
        processed = image_bgr.copy()

        # CLAHE on L-channel
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.contrast_clip, tileGridSize=(8, 8))
        cl_chan = clahe.apply(l_chan)
        enhanced = cv2.merge((cl_chan, a_chan, b_chan))
        processed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Noise reduction
        if self.blur_kernel > 1:
            processed = cv2.GaussianBlur(processed, (self.blur_kernel, self.blur_kernel), 0)

        # Subtle Sharpening
        if self.sharpen:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
            processed = cv2.filter2D(processed, -1, kernel)

        return processed


class MillipedeOBBDetector:
    """Loads and performs inference with the fine-tuned YOLO-OBB millipede segment model."""

    DEFAULT_MODEL_PATHS = [
        "models/millipede_yolov8n_obb.pt",
        "../millicount_dataset/runs/millipede/milli_segment_exp/weights/best.pt",
        "../millicount_dataset/yolov8n-obb.pt"
    ]

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = self._resolve_model_path(model_path)
        if self.model_path and YOLO is not None:
            self.model = YOLO(str(self.model_path))

    def _resolve_model_path(self, model_path):
        if model_path and Path(model_path).exists():
            return Path(model_path).resolve()
        
        base_dir = Path(__file__).resolve().parent
        for candidate in self.DEFAULT_MODEL_PATHS:
            p = (base_dir / candidate).resolve()
            if p.exists():
                return p
        return None

    def is_ready(self):
        return self.model is not None

    def detect(self, image_bgr, conf_thresh=0.25, iou_thresh=0.45):
        """
        Runs YOLO-OBB inference and extracts oriented polygons.
        Returns list of detection dicts:
          class_id, class_name, conf, pts (4x2 np.int32), center (np.float32)
        """
        if not self.is_ready():
            raise RuntimeError("YOLO model weights are not loaded. Train or supply valid weights.")

        results = self.model.predict(
            image_bgr,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False
        )[0]

        detections = []

        # 1. Check for Oriented Bounding Boxes (YOLO-OBB)
        if hasattr(results, "obb") and results.obb is not None and len(results.obb) > 0:
            for obb in results.obb:
                cid = int(obb.cls[0].item())
                cf = float(obb.conf[0].item())
                poly = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)
                center = np.mean(poly, axis=0)
                meta = CLASS_METADATA.get(cid, {"name": f"Class {cid}"})
                detections.append({
                    "class_id": cid,
                    "class_name": meta["name"],
                    "conf": cf,
                    "pts": poly,
                    "center": center
                })

        # 2. Fallback to standard Boxes if model is standard YOLO-Detect
        elif hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cid = int(box.cls[0].item())
                cf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                meta = CLASS_METADATA.get(cid, {"name": f"Class {cid}"})
                detections.append({
                    "class_id": cid,
                    "class_name": meta["name"],
                    "conf": cf,
                    "pts": poly,
                    "center": center
                })

        return detections


class HumanDetector:
    """
    Detects human beings / persons (handlers, bystanders, or hands) using standard COCO YOLO weights.
    Used to distinguish human subjects from millipede specimens.
    """

    DEFAULT_MODEL_PATHS = [
        "yolov8n-seg.pt",
        "yolov8n.pt",
        "models/yolov8n-seg.pt",
        "models/yolov8n.pt"
    ]

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = self._resolve_model_path(model_path)
        if self.model_path and YOLO is not None:
            self.model = YOLO(str(self.model_path))

    def _resolve_model_path(self, model_path):
        if model_path and Path(model_path).exists():
            return Path(model_path).resolve()
        base_dir = Path(__file__).resolve().parent
        for candidate in self.DEFAULT_MODEL_PATHS:
            p = (base_dir / candidate).resolve()
            if p.exists():
                return p
        return None

    def is_ready(self):
        return self.model is not None

    def detect(self, image_bgr, conf_thresh=0.35):
        """
        Detects human beings (COCO class 0: person).
        Returns list of dicts:
          class_id=0, class_name='Human Being', conf, box=[x1,y1,x2,y2], mask (optional)
        """
        if not self.is_ready():
            return []

        results = self.model.predict(
            image_bgr,
            classes=[0],
            conf=conf_thresh,
            verbose=False
        )[0]

        humans = []
        if results.boxes is not None and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                cf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                mask = None
                if results.masks is not None and len(results.masks) > i and results.masks.data is not None:
                    m = results.masks.data[i].cpu().numpy()
                    m = (m > 0.5).astype(np.uint8) * 255
                    mask = cv2.resize(
                        m,
                        (image_bgr.shape[1], image_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                humans.append({
                    "class_id": 0,
                    "class_name": "Human Being",
                    "conf": cf,
                    "box": [x1, y1, x2, y2],
                    "mask": mask
                })
        return humans


def sort_segments_along_spine(detections):
    """
    Sorts detected items in anatomical order from head to telson.
    Uses nearest-neighbor trajectory following with curvature smoothing.
    """
    if not detections:
        return []

    # 1. Identify starting point (Head candidate or principal extremity)
    head_candidates = [d for d in detections if d["class_id"] == 1]
    if head_candidates:
        head_candidates.sort(key=lambda x: x["conf"], reverse=True)
        start_node = head_candidates[0]
    else:
        # Fallback: find extremity along the major principal axis
        centers = np.array([d["center"] for d in detections])
        if len(centers) > 1:
            cov = np.cov(centers, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            major_axis = eigenvectors[:, -1]
            projections = np.dot(centers, major_axis)
            start_idx = int(np.argmin(projections))
            start_node = detections[start_idx]
        else:
            start_node = detections[0]

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


SpineOrderer = sort_segments_along_spine


def analyze_spacing_and_gaps(ordered_detections, gap_ratio=1.85):
    """
    Analyzes inter-segment Euclidean distances along the spine to detect occluded or missed segments.
    """
    if len(ordered_detections) < 3:
        return {
            "gaps": [],
            "median_distance": 0.0,
            "mean_distance": 0.0,
            "std_distance": 0.0,
            "estimated_missing_sum": 0,
            "distances": []
        }

    centers = np.array([d["center"] for d in ordered_detections])
    diffs = np.diff(centers, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    median_dist = float(np.median(dists))
    mean_dist = float(np.mean(dists))
    std_dist = float(np.std(dists))

    gaps = []
    for i, d in enumerate(dists):
        # Gap anomaly if spacing > 1.85 * median spacing
        if d > (median_dist * gap_ratio) and median_dist > 1e-3:
            estimated_missing = int(round((d - median_dist) / median_dist))
            gaps.append({
                "between_indices": (i + 1, i + 2),
                "pt1": centers[i],
                "pt2": centers[i + 1],
                "distance": float(d),
                "median_distance": median_dist,
                "estimated_missing": max(1, estimated_missing)
            })

    missing_sum = sum(g["estimated_missing"] for g in gaps)

    return {
        "gaps": gaps,
        "median_distance": median_dist,
        "mean_distance": mean_dist,
        "std_distance": std_dist,
        "estimated_missing_sum": missing_sum,
        "distances": dists.tolist()
    }


def render_annotated_dashboard(image_bgr, ordered_detections, gap_analysis, human_detections=None, show_badges=True, show_spine=True):
    """
    Renders an elegant, high-contrast visualization with:
    - Rotated oriented bounding boxes for millipedes colored by class.
    - Cyan bounding boxes and badges for detected human beings / handlers.
    - Central body spine polyline.
    - Sequential ring index badges.
    - Gap highlight lines with missing segment estimations.
    - Top executive analytics banner.
    """
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    # 1. Render Human Beings (cyan/sky blue) if detected
    if human_detections:
        for idx, h_det in enumerate(human_detections, start=1):
            box = h_det["box"]
            cf = h_det["conf"]
            mask = h_det.get("mask")

            # Semi-transparent tint on human mask if available
            if mask is not None:
                color_mask = np.zeros_like(vis)
                color_mask[mask > 0] = (255, 180, 0)  # Cyan/Sky blue BGR
                cv2.addWeighted(color_mask, 0.25, vis, 1.0, 0, vis)

            # Bounding box
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 190, 0), 2, lineType=cv2.LINE_AA)
            tag = f"Human Being ({cf:.0%})"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (box[0], max(0, box[1] - th - 6)), (box[0] + tw + 6, max(th + 6, box[1])), (20, 20, 25), -1)
            cv2.rectangle(vis, (box[0], max(0, box[1] - th - 6)), (box[0] + tw + 6, max(th + 6, box[1])), (255, 190, 0), 1)
            cv2.putText(vis, tag, (box[0] + 3, max(th + 2, box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 150), 1, cv2.LINE_AA)

    # 2. Draw continuous millipede body spine
    if show_spine and len(ordered_detections) > 1:
        spine_pts = np.array([d["center"].astype(np.int32) for d in ordered_detections])
        cv2.polylines(vis, [spine_pts], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 3. Highlight gaps
    for gap in gap_analysis.get("gaps", []):
        p1 = tuple(gap["pt1"].astype(int))
        p2 = tuple(gap["pt2"].astype(int))
        cv2.line(vis, p1, p2, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        gap_tag = f"Gap: ~+{gap['estimated_missing']}"
        cv2.putText(vis, gap_tag, (mid[0] + 5, mid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, gap_tag, (mid[0] + 5, mid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Draw millipede segments
    total_count = len(ordered_detections)
    for idx, d in enumerate(ordered_detections, start=1):
        cid = d["class_id"]
        meta = CLASS_METADATA.get(cid, {"color_bgr": (255, 255, 255)})
        color = meta["color_bgr"]
        pts = d["pts"]

        # Polygon box
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Center marker
        c = tuple(d["center"].astype(int))
        cv2.circle(vis, c, 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, c, 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # Sequential badges
        if show_badges:
            is_endpoint = (idx == 1 or idx == total_count)
            is_interval = (idx % 3 == 0) or (total_count <= 25)
            if is_endpoint or is_interval:
                tag = f"{idx}"
                if cid == 1:
                    tag = f"1:Head"
                elif cid == 2:
                    tag = f"{idx}:Telson"

                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                bx, by = c[0] + 6, c[1] - 4
                cv2.rectangle(vis, (bx - 2, by - th - 3), (bx + tw + 2, by + 3), (20, 20, 20), -1)
                cv2.rectangle(vis, (bx - 2, by - th - 3), (bx + tw + 2, by + 3), color, 1)
                cv2.putText(vis, tag, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Top Analytics Banner
    head_count = sum(1 for d in ordered_detections if d["class_id"] == 1)
    body_count = sum(1 for d in ordered_detections if d["class_id"] == 0)
    telson_count = sum(1 for d in ordered_detections if d["class_id"] == 2)
    missing_count = gap_analysis.get("estimated_missing_sum", 0)
    est_true_total = total_count + missing_count
    human_count = len(human_detections) if human_detections else 0

    banner_height = 54
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (22, 22, 28)

    if human_count > 0:
        title_text = f"No millipede detected (Human Being Detected: {human_count})"
        cv2.putText(banner, title_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 140, 255), 2, cv2.LINE_AA)
        stats_text = f"Human subject detected in frame ({human_count}) -- Millipede analysis suppressed"
        cv2.putText(banner, stats_text, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        title_text = f"Millipede Rings: {total_count}"
        if missing_count > 0:
            title_text += f" | Est. True Total: {est_true_total} (+{missing_count} occluded)"
        cv2.putText(banner, title_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 245, 170), 2, cv2.LINE_AA)
        stats_text = f"Head: {head_count}  |  Body Annuli: {body_count}  |  Telson: {telson_count}  |  Gaps: {len(gap_analysis.get('gaps', []))}"
        cv2.putText(banner, stats_text, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    dashboard = np.vstack([banner, vis])
    return dashboard


class ClassicalProfiler:
    """
    Classical Morphological / Cross-Sectional Profiler.
    Kept for baseline comparison and scientific profiling.
    """
    def __init__(self, contrast_clip=2.5):
        self.contrast_clip = contrast_clip

    def segment_mask(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest, 255, 0).astype(np.uint8)
            return mask
        return None

    def extract_skeleton_and_unroll(self, image_bgr, mask, half_width=30):
        try:
            skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, skeleton = cv2.threshold(dist, 0.35 * dist.max(), 255, cv2.THRESH_BINARY)
            skeleton = skeleton.astype(np.uint8)

        ys, xs = np.where(skeleton > 0)
        if len(xs) < 15:
            return None, None, 0

        pts = np.column_stack((xs, ys))
        sorted_pts = [pts[0]]
        unvisited = pts[1:].tolist()
        while unvisited:
            cur = sorted_pts[-1]
            dists = [np.linalg.norm(np.array(p) - cur) for p in unvisited]
            idx = int(np.argmin(dists))
            if dists[idx] > 20:
                break
            sorted_pts.append(np.array(unvisited.pop(idx)))

        centerline = np.array(sorted_pts, dtype=np.float32)
        if len(centerline) < 15:
            return None, None, 0

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        n = len(centerline)
        unrolled = np.zeros((half_width * 2 + 1, n), dtype=np.uint8)

        for i in range(1, n - 1):
            tangent = centerline[i + 1] - centerline[i - 1]
            norm = np.linalg.norm(tangent)
            if norm < 1e-6:
                continue
            normal = np.array([-tangent[1], tangent[0]]) / norm
            center = centerline[i]

            for j, d in enumerate(range(-half_width, half_width + 1)):
                sample = center + normal * d
                x, y = int(round(sample[0])), int(round(sample[1]))
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    unrolled[j, i] = gray[y, x]

        # 1D signal extraction
        brightness = np.mean(unrolled, axis=0)
        gradient = np.abs(np.gradient(brightness))
        combined_signal = gaussian_filter1d(gradient, sigma=2)
        denom = (combined_signal.max() - combined_signal.min()) + 1e-6
        combined_signal = (combined_signal - combined_signal.min()) / denom

        peaks, _ = find_peaks(combined_signal, distance=max(4, int(n * 0.015)), prominence=np.std(combined_signal) * 0.4)
        return unrolled, combined_signal, len(peaks)
