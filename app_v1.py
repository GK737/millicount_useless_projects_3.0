import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from scipy.signal import find_peaks
from ultralytics import YOLO

# ==========================================
# 1. MODULAR PREPROCESSING PIPELINE
# ==========================================
class ImagePreprocessor:
    def __init__(self, target_size=(640, 640), blur_kernel=5, contrast_clip=2.0, sharpen=True):
        self.target_size = target_size
        self.blur_kernel = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
        self.contrast_clip = contrast_clip
        self.sharpen = sharpen

    def process(self, image_bgr):
        # Resizing
        h, w = image_bgr.shape[:2]
        processed = cv2.resize(image_bgr, self.target_size, interpolation=cv2.INTER_AREA)

        # Color Space Conversion & Contrast Enhancement (CLAHE on L-channel)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=self.contrast_clip, tileGridSize=(8, 8))
        cl_channel = clahe.apply(l_channel)
        
        enhanced = cv2.merge((cl_channel, a_channel, b_channel))
        processed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Noise Reduction
        if self.blur_kernel > 1:
            processed = cv2.GaussianBlur(processed, (self.blur_kernel, self.blur_kernel), 0)

        # Optional Sharpening
        if self.sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed = cv2.filter2D(processed, -1, kernel)

        return processed, (w, h)

# ==========================================
# 2. SEGMENTATION & AXIAL COUNTING ENGINE
# ==========================================
class MillipedeAnalyzer:
    def __init__(self, model_weight="yolov8n-seg.pt"):
        # Load custom fine-tuned YOLO model (or standard weights as baseline)
        self.model = YOLO(model_weight)

    def analyze(self, image_bgr):
        # Step 1: Run Instance Segmentation
        results = self.model(image_bgr, verbose=False)[0]
        
        if results.masks is None or len(results.masks) == 0:
            return image_bgr, 0, 0, None, "No millipede detected in frame."

        # Extract primary instance mask (highest confidence score)
        mask = results.masks.data[0].cpu().numpy().astype(np.uint8) * 255
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))

        # Step 2: Skeletonization (Centerline extraction)
        # Fallback to distance transform if cv2.ximgproc is unavailable
        try:
            skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except AttributeError:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, skeleton = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
            skeleton = skeleton.astype(np.uint8)

        # Step 3: Extract & Order Body Axis Coordinates
        y_pts, x_pts = np.where(skeleton > 0)
        if len(x_pts) < 15:
            return image_bgr, 0, 0, None, "Millipede mask too small or fragmented."

        # Sort coordinates linearly along the major gradient vector to handle bends
        pts = np.column_stack((x_pts, y_pts))
        sorted_indices = np.argsort(pts[:, 0])  # Primary sorting on X-axis
        sorted_pts = pts[sorted_indices]

        # Step 4: Perpendicular Sampling for Ring Boundaries
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        profile_signal = []

        for i in range(2, len(sorted_pts) - 2):
            pt = sorted_pts[i]
            prev_pt = sorted_pts[i - 2]
            next_pt = sorted_pts[i + 2]

            # Compute tangential and normal (perpendicular) vectors
            tangent = next_pt - prev_pt
            norm = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            norm_len = np.linalg.norm(norm)
            if norm_len == 0:
                continue
            norm /= norm_len

            # Sample orthogonal strip across the segment boundary
            sample_pts = [pt + (norm * d).astype(int) for d in range(-4, 5)]
            sample_vals = []
            for sp in sample_pts:
                if 0 <= sp[0] < gray.shape[1] and 0 <= sp[1] < gray.shape[0]:
                    sample_vals.append(gray[sp[1], sp[0]])
            
            if sample_vals:
                profile_signal.append(np.mean(sample_vals))

        profile_signal = np.array(profile_signal)
        
        # Step 5: Peak Detection & Segment Counting
        peaks, _ = find_peaks(profile_signal, distance=4, prominence=8)
        segment_count = len(peaks)

        # Step 6: Leg Estimation Formula
        # Total Legs = 6 + [4 x (Total Segments - 5)]
        if segment_count >= 5:
            estimated_legs = 6 + (4 * (segment_count - 5))
        else:
            estimated_legs = 0

        # Step 7: Visualization Overlay
        annotated = image_bgr.copy()
        
        # Overlay Spine
        for pt in sorted_pts:
            cv2.circle(annotated, tuple(pt), 1, (255, 0, 0), -1)

        # Overlay Segment Boundaries (Peaks)
        for peak_idx in peaks:
            if peak_idx < len(sorted_pts):
                peak_pt = tuple(sorted_pts[peak_idx])
                cv2.circle(annotated, peak_pt, 4, (0, 255, 0), -1)
                cv2.drawMarker(annotated, peak_pt, (0, 0, 255), cv2.MARKER_CROSS, 8, 2)

        return annotated, segment_count, estimated_legs, profile_signal, None

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="MILLICOUNT", layout="wide", page_icon="🐛")

st.title("🐛 MILLICOUNT")
st.caption("A very USEFULL tool for counting millipede body segments and estimating leg counts.")

# Sidebar - Modular Preprocessing Controls
st.sidebar.header("⚙️ Pipeline Preprocessing")
blur_val = st.sidebar.slider("Gaussian Blur Kernel", 1, 15, 5, step=2)
contrast_val = st.sidebar.slider("CLAHE Clip Limit", 1.0, 10.0, 2.5, step=0.5)
enable_sharpen = st.sidebar.checkbox("Enable Edge Sharpening", True)

preprocessor = ImagePreprocessor(blur_kernel=blur_val, contrast_clip=contrast_val, sharpen=enable_sharpen)
analyzer = MillipedeAnalyzer()

# Input Mode Switcher
mode = st.radio("Select Input Mode", ["A. Image Mode", "B. Live Camera Mode"], horizontal=True)

if mode == "A. Image Mode":
    uploaded_file = st.file_uploader("Upload Millipede Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Preprocessed Input")
            prep_img, _ = preprocessor.process(raw_bgr)
            st.image(cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("🔬 Analyze Image", use_container_width=True):
            annotated_img, seg_count, leg_count, signal, err = analyzer.analyze(prep_img)
            
            if err:
                st.error(err)
            else:
                with col2:
                    st.subheader("2. Detected Boundaries & Centerline")
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                # Metrics Section
                st.divider()
                m1, m2 = st.columns(2)
                m1.metric("Counted Body Segments", f"{seg_count}")
                m2.metric(
                    "Estimated Total Legs", 
                    f"{leg_count}", 
                    help="Calculated via bio-formula: 6 + [4 × (Segments - 5)]. Derived from segment count."
                )
                
                # Boundary Visualization Chart
                if signal is not None and len(signal) > 0:
                    st.subheader("3. Longitudinal Intensity Profile (Segment Boundaries)")
                    fig, ax = plt.subplots(figsize=(10, 2.5))
                    ax.plot(signal, color="#4CAF50", linewidth=1.5, label="Axial Gradient")
                    ax.set_ylabel("Pixel Intensity")
                    ax.set_xlabel("Body Axis Traversal Index")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig)

elif mode == "B. Live Camera Mode":
    st.info("Capture a frame from the live webcam feed to freeze and execute segment analysis.")
    
    # Built-in Streamlit camera capture tool (uses web API safely)
    camera_image = st.camera_input("Live Webcam Feed")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        prep_img, _ = preprocessor.process(cv2_img)
        annotated_img, seg_count, leg_count, signal, err = analyzer.analyze(prep_img)

        if err:
            st.warning(err)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Analyzed Frozen Frame")
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.subheader("Estimation Summary")
                st.metric("Total Segments", seg_count)
                st.metric("Estimated Leg Count", leg_count)
                st.caption("⚠️ **Note:** Leg total is an *estimated value* calculated from segment counts using biological diplosegment math.")