import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from scipy.signal import find_peaks

# ==========================================
# 1. MODULAR PREPROCESSING PIPELINE
# ==========================================
class ImagePreprocessor:
    def __init__(self, blur_kernel=5, contrast_clip=3.0, sharpen=True):
        self.blur_kernel = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
        self.contrast_clip = contrast_clip
        self.sharpen = sharpen

    def process(self, image_bgr):
        # 1. Contrast Enhancement via CLAHE in LAB space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.contrast_clip, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 2. Noise Reduction
        if self.blur_kernel > 1:
            enhanced = cv2.GaussianBlur(enhanced, (self.blur_kernel, self.blur_kernel), 0)

        # 3. Optional Sharpening
        if self.sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

# ==========================================
# 2. SKELETON & SEGMENTATION ENGINE (NO ML)
# ==========================================
class ClassicalMillipedeAnalyzer:
    def segment_body(self, image_bgr, min_area=1000):
        """Isolate millipede body using Otsu thresholding & Morphological operations."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Otsu thresholding after subtle blurring
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean background noise using Morphological Closing and Opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find largest elongated contour
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < min_area:
            return None, None

        # Mask out everything except the largest body contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        return mask, gray

    def analyze(self, image_bgr, peak_distance=4, peak_prominence=10):
        mask, gray = self.segment_body(image_bgr)
        if mask is None:
            return image_bgr, 0, 0, None, "Could not isolate a distinct millipede body from the background."

        # Step 1: Skeletonization (Centerline extraction)
        try:
            skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, skeleton = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
            skeleton = skeleton.astype(np.uint8)

        # Step 2: Extract spine coordinates
        y_pts, x_pts = np.where(skeleton > 0)
        if len(x_pts) < 15:
            return image_bgr, 0, 0, None, "Millipede body axis too small or disconnected."

        # Step 3: Order coordinates from head to tail using nearest-neighbor sorting
        pts = np.column_stack((x_pts, y_pts))
        sorted_pts = [pts[0]]
        remaining = pts[1:].tolist()

        while remaining:
            last_pt = sorted_pts[-1]
            # Find nearest point along the skeleton
            distances = [np.linalg.norm(np.array(p) - last_pt) for p in remaining]
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] > 15:  # Break if skeleton has a large break
                break
            sorted_pts.append(np.array(remaining.pop(nearest_idx)))

        sorted_pts = np.array(sorted_pts)

        # Step 4: Sample intensity along orthogonal vectors (rings)
        profile_signal = []
        for i in range(2, len(sorted_pts) - 2):
            pt = sorted_pts[i]
            prev_pt = sorted_pts[i - 2]
            next_pt = sorted_pts[i + 2]

            tangent = next_pt - prev_pt
            norm = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            norm_len = np.linalg.norm(norm)
            if norm_len == 0:
                continue
            norm /= norm_len

            # Sample 9 pixels across the width perpendicular to the spine
            sample_pts = [pt + (norm * d).astype(int) for d in range(-4, 5)]
            sample_vals = [
                gray[sp[1], sp[0]] for sp in sample_pts
                if 0 <= sp[0] < gray.shape[1] and 0 <= sp[1] < gray.shape[0]
            ]

            if sample_vals:
                profile_signal.append(np.mean(sample_vals))

        profile_signal = np.array(profile_signal)

        # Step 5: Peak detection for rings
        peaks, _ = find_peaks(profile_signal, distance=peak_distance, prominence=peak_prominence)
        segment_count = len(peaks)

        # Step 6: Leg estimation formula
        # Total Legs = 6 + [4 x (Total Segments - 5)]
        estimated_legs = 6 + (4 * (segment_count - 5)) if segment_count >= 5 else 0

        # Step 7: Drawing overlays
        annotated = image_bgr.copy()
        
        # Draw skeleton body centerline
        for pt in sorted_pts:
            cv2.circle(annotated, tuple(pt), 1, (255, 0, 0), -1)

        # Draw detected segment boundaries (Peaks)
        for peak_idx in peaks:
            if peak_idx < len(sorted_pts):
                peak_pt = tuple(sorted_pts[peak_idx])
                cv2.circle(annotated, peak_pt, 4, (0, 255, 0), -1)
                cv2.drawMarker(annotated, peak_pt, (0, 0, 255), cv2.MARKER_CROSS, 6, 2)

        return annotated, segment_count, estimated_legs, profile_signal, None

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="millicount", layout="wide", page_icon="🐛")

st.title("🐛 millicount")
st.caption("Computer-Vision Segment Analysis & Leg Estimation Engine powered by millicount_dataset")

# Sidebar Engine & Preprocessing Controls
st.sidebar.header("⚙️ Counting Engine")
engine_choice = st.sidebar.radio(
    "Select Model Engine",
    [
        "🎯 YOLO-OBB (High Accuracy)",
        "🔬 Classical Pipeline (Morphology)"
    ]
)

st.sidebar.header("⚙️ Image Preprocessing")
blur_val = st.sidebar.slider("Gaussian Blur Kernel", 1, 15, 5, step=2)
contrast_val = st.sidebar.slider("CLAHE Contrast Limit", 1.0, 10.0, 3.0, step=0.5)
enable_sharpen = st.sidebar.checkbox("Enable Sharpening", True)

# Default parameters
conf_thresh = 0.20
gap_ratio = 1.85
peak_dist = 4
peak_prom = 10

if engine_choice.startswith("🎯"):
    st.sidebar.header("🎯 YOLO-OBB Fine-Tuning")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.90, 0.20, 0.05)
    gap_ratio = st.sidebar.slider("Gap Anomaly Ratio", 1.3, 2.8, 1.85, 0.05)
else:
    st.sidebar.header("🔬 Peak Detection Fine-Tuning")
    peak_dist = st.sidebar.slider("Min Peak Distance (px)", 1, 15, 4)
    peak_prom = st.sidebar.slider("Peak Prominence", 1, 30, 10)

preprocessor = ImagePreprocessor(blur_kernel=blur_val, contrast_clip=contrast_val, sharpen=enable_sharpen)
analyzer = ClassicalMillipedeAnalyzer()

# Lazy load YOLO detector if needed
obb_detector = None
if engine_choice.startswith("🎯"):
    try:
        from engine import MillipedeOBBDetector, sort_segments_along_spine, analyze_spacing_and_gaps, render_annotated_dashboard
        obb_detector = MillipedeOBBDetector("models/millipede_yolov8n_obb.pt")
    except Exception as e:
        st.sidebar.warning(f"Could not load YOLO engine: {e}")

mode = st.radio("Select Input Mode", ["A. Image Mode", "B. Live Camera Mode"], horizontal=True)

if mode == "A. Image Mode":
    uploaded_file = st.file_uploader("Upload Millipede Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Preprocessed Image")
            prep_img = preprocessor.process(raw_bgr)
            st.image(cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("🔬 Analyze Image", use_container_width=True):
            if engine_choice.startswith("🎯") and obb_detector and obb_detector.is_ready():
                dets = obb_detector.detect(prep_img, conf_thresh=conf_thresh)
                ordered = sort_segments_along_spine(dets)
                gaps = analyze_spacing_and_gaps(ordered, gap_ratio=gap_ratio)
                seg_count = len(ordered)
                missing = gaps["estimated_missing_sum"]
                total_est = seg_count + missing
                leg_count = 6 + (4 * (total_est - 5)) if total_est >= 5 else 0
                annotated_img = render_annotated_dashboard(prep_img, ordered, gaps)

                with col2:
                    st.subheader("2. YOLO-OBB Segment Ring Overlay")
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Detected Rings", f"{seg_count}")
                m2.metric("Estimated Total Rings", f"{total_est}", delta=f"+{missing} occluded" if missing > 0 else "Continuous")
                m3.metric(
                    "Estimated Total Legs", 
                    f"{leg_count}", 
                    help="Formula: 6 + [4 × (Segments - 5)]. Estimated from detected segment boundaries."
                )

            else:
                annotated_img, seg_count, leg_count, signal, err = analyzer.analyze(
                    prep_img, peak_distance=peak_dist, peak_prominence=peak_prom
                )
                
                if err:
                    st.error(err)
                else:
                    with col2:
                        st.subheader("2. Detected Axis & Segment Ring Overlay")
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                    st.divider()
                    m1, m2 = st.columns(2)
                    m1.metric("Detected Body Segments", f"{seg_count}")
                    m2.metric(
                        "Estimated Total Legs", 
                        f"{leg_count}", 
                        help="Formula: 6 + [4 × (Segments - 5)]. Estimated from detected segment boundaries."
                    )
                    
                    if signal is not None and len(signal) > 0:
                        st.subheader("3. Longitudinal Intensity Profile (Peak Graph)")
                        fig, ax = plt.subplots(figsize=(10, 2.5))
                        ax.plot(signal, color="#2E7D32", linewidth=1.5, label="Axial Gradient")
                        ax.set_ylabel("Pixel Intensity")
                        ax.set_xlabel("Body Centerline Step Index")
                        ax.grid(True, linestyle="--", alpha=0.5)
                        st.pyplot(fig)

elif mode == "B. Live Camera Mode":
    st.info("Capture a live frame from your webcam to run segment detection.")
    camera_image = st.camera_input("Live Webcam Feed")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        prep_img = preprocessor.process(cv2_img)

        if engine_choice.startswith("🎯") and obb_detector and obb_detector.is_ready():
            dets = obb_detector.detect(prep_img, conf_thresh=conf_thresh)
            ordered = sort_segments_along_spine(dets)
            gaps = analyze_spacing_and_gaps(ordered, gap_ratio=gap_ratio)
            seg_count = len(ordered)
            missing = gaps["estimated_missing_sum"]
            total_est = seg_count + missing
            leg_count = 6 + (4 * (total_est - 5)) if total_est >= 5 else 0
            annotated_img = render_annotated_dashboard(prep_img, ordered, gaps)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Analyzed Frame (YOLO-OBB)")
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("Results")
                st.metric("Detected Rings", f"{seg_count}")
                st.metric("Estimated Total Rings", f"{total_est}", delta=f"+{missing} occluded" if missing > 0 else "Continuous")
                st.metric("Estimated Total Legs", f"{leg_count}", help="Formula: 6 + [4 × (Segments - 5)].")

        else:
            annotated_img, seg_count, leg_count, signal, err = analyzer.analyze(
                prep_img, peak_distance=peak_dist, peak_prominence=peak_prom
            )

            if err:
                st.warning(err)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Analyzed Frame")
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.subheader("Results")
                    st.metric("Total Segments Detected", seg_count)
                    st.metric("Estimated Total Legs", leg_count)
                    st.caption("⚠️ **Note:** Leg count is an estimated calculation based on detected segments.")