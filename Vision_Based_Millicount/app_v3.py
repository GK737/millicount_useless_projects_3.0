"""
🐛 MILLICOUNT - High-Accuracy Millipede Segment & Ring Counter
Empowered by YOLO-OBB Deep Learning, Spine Tracking, and Occlusion Analysis.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from engine import (
    MillipedeOBBDetector,
    HumanDetector,
    SpineOrderer,
    analyze_spacing_and_gaps,
    render_annotated_dashboard,
    ImagePreprocessor,
    ClassicalProfiler,
    CLASS_METADATA
)


# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="MILLICOUNT - Segment & Ring Counter",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for KPI cards, tags, and visuals
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #2e3240;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f0b4;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-head {
        background-color: rgba(255, 165, 0, 0.2);
        color: #ffa500;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-body {
        background-color: rgba(0, 235, 120, 0.2);
        color: #00eb78;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-telson {
        background-color: rgba(210, 50, 255, 0.2);
        color: #d232ff;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MODEL CACHING & ENGINE INITIALIZATION
# ==============================================================================
@st.cache_resource(show_spinner="Loading YOLO-OBB Millipede Detection Model...")
def load_detector():
    detector = MillipedeOBBDetector()
    return detector


@st.cache_resource(show_spinner="Loading Human Subject Detection Model...")
def load_human_detector():
    return HumanDetector()


detector = load_detector()
human_detector = load_human_detector()
preprocessor = ImagePreprocessor()
classical_engine = ClassicalProfiler()


# ==============================================================================
# 3. SIDEBAR CONTROLS & HYPERPARAMETERS
# ==============================================================================
st.sidebar.title("⚙️ Analysis Pipeline")

pipeline_mode = st.sidebar.radio(
    "Counting Engine Mode",
    [
        "🎯 YOLO-OBB Deep Learning (High Accuracy)",
        "🔬 Classical Signal Profiler (Morphology)"
    ]
)

st.sidebar.divider()

if pipeline_mode.startswith("🎯"):
    st.sidebar.subheader("🐛 Millipede YOLO-OBB")
    conf_thresh = st.sidebar.slider(
        "Segment Confidence",
        min_value=0.05,
        max_value=0.90,
        value=0.20,
        step=0.05,
        help="Lower values detect faint segments; higher values filter background noise."
    )
    iou_thresh = st.sidebar.slider(
        "NMS IoU Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression overlap threshold."
    )
    gap_ratio = st.sidebar.slider(
        "Gap Anomaly Multiplier",
        min_value=1.3,
        max_value=2.8,
        value=1.85,
        step=0.05,
        help="Flag missing segments if distance between adjacent rings > ratio * median."
    )
    show_badges = st.sidebar.checkbox("Show Sequential Ring Badges", value=True)
    show_spine = st.sidebar.checkbox("Show Body Spine Trajectory", value=True)

    st.sidebar.divider()
    st.sidebar.subheader("👤 Human vs Millipede Detection")
    detect_humans = st.sidebar.checkbox(
        "Detect Human Being / Handler",
        value=True,
        help="Differentiates human handlers and bystanders from millipede specimens."
    )
    human_conf = st.sidebar.slider(
        "Human Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Confidence threshold for identifying human beings in frame."
    )
else:
    detect_humans = False
    human_conf = 0.35

with st.sidebar.expander("🛠️ Advanced Image Preprocessing", expanded=False):
    enable_prep = st.checkbox("Enable Preprocessing Filter", value=True)
    contrast_clip = st.slider("CLAHE Contrast Limit", 1.0, 5.0, 2.0, 0.5)
    blur_kernel = st.slider("Gaussian Blur Kernel", 1, 9, 3, 2)
    enable_sharpen = st.checkbox("Edge Sharpening", value=True)


# ==============================================================================
# 4. MAIN INTERFACE HEADER
# ==============================================================================
st.title("🐛 MILLICOUNT")
st.markdown(
    "Automated deep-learning vision pipeline for **anatomical identification, sequential ring counting, "
    "and occlusion analysis** of millipedes (*Class Diplopoda*)."
)

# Input Source Selection
input_source = st.radio(
    "Select Image Source",
    [
        "📸 Upload Image",
        "🧪 Sample Specimen (millipede.jpg)",
        "📷 Live Webcam Feed"
    ],
    horizontal=True
)

raw_bgr = None

if input_source == "📸 Upload Image":
    uploaded_file = st.file_uploader(
        "Upload millipede photograph...",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif input_source == "🧪 Sample Specimen (millipede.jpg)":
    sample_path = Path("millipede.jpg")
    if sample_path.exists():
        raw_bgr = cv2.imread(str(sample_path))
        st.info("Loaded default local benchmark: `millipede.jpg`")
    else:
        st.warning("`millipede.jpg` not found in workspace.")

elif input_source == "📷 Live Webcam Feed":
    camera_image = st.camera_input("Take a snapshot of specimen")
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
        raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# ==============================================================================
# 5. EXECUTE ANALYSIS PIPELINE
# ==============================================================================
if raw_bgr is not None:
    # Optional image preprocessing
    if enable_prep:
        proc = ImagePreprocessor(
            blur_kernel=blur_kernel,
            contrast_clip=contrast_clip,
            sharpen=enable_sharpen
        )
        work_img = proc.process(raw_bgr)
    else:
        work_img = raw_bgr.copy()

    # Layout: Input vs Result
    st.divider()

    # --------------------------------------------------------------------------
    # MODE A: DEEP LEARNING YOLO-OBB ENGINE
    # --------------------------------------------------------------------------
    if pipeline_mode.startswith("🎯"):
        if not detector.is_ready():
            st.error(
                "❌ YOLO-OBB Model weights not found! Please ensure training has finished or "
                "`models/millipede_yolov8n_obb.pt` is present."
            )
        else:
            with st.spinner("Scanning image for subjects..."):
                # 1. Detect Human Beings first if enabled
                human_detections = []
                if detect_humans and human_detector and human_detector.is_ready():
                    human_detections = human_detector.detect(work_img, conf_thresh=human_conf)

                num_humans = len(human_detections)

                # RULE: If a human is detected -> No millipede detected!
                if num_humans > 0:
                    ordered_detections = []
                    gap_data = {
                        "gaps": [],
                        "median_distance": 0.0,
                        "estimated_missing_sum": 0,
                        "distances": []
                    }
                    dashboard_img = render_annotated_dashboard(
                        work_img,
                        ordered_detections=[],
                        gap_analysis=gap_data,
                        human_detections=human_detections,
                        show_badges=False,
                        show_spine=False
                    )
                else:
                    # 2. No human in frame -> Proceed to detect Millipede Segments & Anatomy
                    raw_detections = detector.detect(
                        work_img,
                        conf_thresh=conf_thresh,
                        iou_thresh=iou_thresh
                    )
                    ordered_detections = SpineOrderer(raw_detections)
                    gap_data = analyze_spacing_and_gaps(ordered_detections, gap_ratio=gap_ratio)
                    dashboard_img = render_annotated_dashboard(
                        work_img,
                        ordered_detections,
                        gap_data,
                        human_detections=[],
                        show_badges=show_badges,
                        show_spine=show_spine
                    )

            total_detected = len(ordered_detections)
            head_count = sum(1 for d in ordered_detections if d["class_id"] == 1)
            body_count = sum(1 for d in ordered_detections if d["class_id"] == 0)
            telson_count = sum(1 for d in ordered_detections if d["class_id"] == 2)
            missing_count = gap_data.get("estimated_missing_sum", 0)
            est_true_total = total_detected + missing_count

            # --- SUBJECT CLASSIFICATION & DIAGNOSTIC BANNER ---
            if num_humans > 0:
                st.error(
                    f"❌ **No millipede detected.** (Human being detected in frame: {num_humans} person(s))"
                )
            elif total_detected > 0:
                st.success(
                    f"🐛 **Millipede Specimen Verified**: {total_detected} anatomical rings detected (0 humans in frame)."
                )
            else:
                st.warning(
                    "⚠️ **No millipede detected.**"
                )

            # --- METRICS ROW ---
            if num_humans > 0:
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    st.metric(
                        label="Millipede Detection Status",
                        value="No millipede detected",
                        delta="Human in frame",
                        delta_color="inverse"
                    )
                with kpi2:
                    st.metric(
                        label="Human Beings in Frame",
                        value=f"{num_humans} Detected",
                        delta="Millipede analysis suppressed",
                        delta_color="off"
                    )
                with kpi3:
                    st.metric(
                        label="Detected Rings",
                        value="0",
                        delta="None",
                        delta_color="off"
                    )

                st.divider()

                st.subheader("🔬 Vision Analysis Frame")
                st.image(
                    cv2.cvtColor(dashboard_img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption="Human Being Detected in frame: Millipede detection suppressed -> No millipede detected."
                )

                with st.expander("👤 Detected Human Subjects Table", expanded=True):
                    human_table = []
                    for idx, h in enumerate(human_detections, start=1):
                        b = h["box"]
                        human_table.append({
                            "Person #": idx,
                            "Subject Type": h["class_name"],
                            "Confidence": f"{h['conf']:.3f}",
                            "Bounding Box [x1, y1, x2, y2]": f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]",
                            "Status": "No millipede detected"
                        })
                    st.dataframe(pd.DataFrame(human_table), use_container_width=True)

            else:
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

                with kpi1:
                    st.metric(
                        label="Detected Rings",
                        value=f"{total_detected}",
                        delta=f"{body_count} Body Annuli"
                    )

                with kpi2:
                    st.metric(
                        label="Estimated True Count",
                        value=f"{est_true_total}",
                        delta=f"+{missing_count} Occluded Rings" if missing_count > 0 else "Continuous",
                        delta_color="normal" if missing_count == 0 else "inverse"
                    )

                with kpi3:
                    est_legs = 6 + (4 * (est_true_total - 5)) if est_true_total >= 5 else 0
                    st.metric(
                        label="Estimated Legs",
                        value=f"{est_legs}",
                        delta="Diplopod formula",
                        delta_color="off",
                        help="Formula: 6 + [4 × (Segments - 5)] based on diplopod anatomy."
                    )

                with kpi4:
                    head_status = "✅ Found" if head_count > 0 else "⚠️ Missing"
                    telson_status = "✅ Found" if telson_count > 0 else "⚠️ Missing"
                    st.metric(
                        label="Anatomy Identified",
                        value=f"Head: {head_count} | Telson: {telson_count}",
                        delta=f"{head_status} / {telson_status}",
                        delta_color="off"
                    )

                with kpi5:
                    num_gaps = len(gap_data.get("gaps", []))
                    st.metric(
                        label="Continuity Quality",
                        value="Continuous" if num_gaps == 0 else f"{num_gaps} Gap(s) Flagged",
                        delta=f"Median: {gap_data.get('median_distance', 0):.1f}px",
                        delta_color="normal" if num_gaps == 0 else "inverse"
                    )

                st.divider()

                # --- VISUALIZATION ROW ---
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    st.subheader("🔬 Annotated Vision Dashboard")
                    st.image(
                        cv2.cvtColor(dashboard_img, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                        caption="Oriented Polygons: Green = Body Annuli, Amber = Head, Magenta = Telson"
                    )

                with col_right:
                    st.subheader("📈 Inter-Segment Spacing Profile")
                    distances = gap_data.get("distances", [])
                    if len(distances) > 0:
                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        x_axis = np.arange(1, len(distances) + 1)
                        med = gap_data["median_distance"]
                        thresh = med * gap_ratio

                        ax.plot(x_axis, distances, color="#00eb78", marker="o", markersize=4, linewidth=1.5, label="Inter-Segment Distance (px)")
                        ax.axhline(med, color="#00f0b4", linestyle="--", linewidth=1.2, label=f"Median Spacing ({med:.1f}px)")
                        ax.axhline(thresh, color="#ff4b4b", linestyle=":", linewidth=1.5, label=f"Gap Threshold ({thresh:.1f}px)")

                        for g in gap_data.get("gaps", []):
                            b1 = g["between_indices"][0]
                            ax.scatter([b1], [g["distance"]], color="#ff0000", s=60, zorder=5)

                        ax.set_facecolor("#1e212b")
                        fig.patch.set_facecolor("#1e212b")
                        ax.tick_params(colors="#dcdcdc")
                        ax.xaxis.label.set_color("#dcdcdc")
                        ax.yaxis.label.set_color("#dcdcdc")
                        ax.set_xlabel("Adjacent Segment Index (# to #+1)")
                        ax.set_ylabel("Distance (pixels)")
                        ax.grid(True, linestyle="--", alpha=0.3, color="#555")
                        ax.legend(facecolor="#2b2e3b", edgecolor="#444", labelcolor="#fff")
                        st.pyplot(fig)
                    else:
                        st.info("At least 3 sequential segments required for spacing profile.")

                    # Detailed Gap Information
                    if len(gap_data.get("gaps", [])) > 0:
                        with st.expander("⚠️ Gap & Occlusion Analysis Details", expanded=True):
                            gap_rows = []
                            for g in gap_data["gaps"]:
                                b1, b2 = g["between_indices"]
                                gap_rows.append({
                                    "Between Segments": f"#{b1} ➔ #{b2}",
                                    "Distance (px)": f"{g['distance']:.1f}",
                                    "Expected (px)": f"{g['median_distance']:.1f}",
                                    "Estimated Missing": f"~{g['estimated_missing']} ring(s)"
                                })
                            st.table(pd.DataFrame(gap_rows))
                    else:
                        st.success("✅ Spine continuity verified: zero occlusions or abnormal ring gaps detected.")

                # --- DETECTION DATA TABLES ---
                with st.expander("📋 Raw Anatomical Detection Table (Millipede)"):
                    if total_detected > 0:
                        table_data = []
                        for idx, d in enumerate(ordered_detections, start=1):
                            table_data.append({
                                "Ring #": idx,
                                "Anatomical Part": d["class_name"],
                                "Confidence": f"{d['conf']:.3f}",
                                "Center (X, Y)": f"({d['center'][0]:.1f}, {d['center'][1]:.1f})"
                            })
                        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                    else:
                        st.info("No millipede segments detected in this frame.")

    # --------------------------------------------------------------------------
    # MODE B: CLASSICAL SIGNAL PROFILER
    # --------------------------------------------------------------------------
    else:
        st.subheader("🔬 Classical Morphological & Cross-Sectional Profiler")
        with st.spinner("Extracting body contour and unrolling transverse signals..."):
            mask = classical_engine.segment_mask(work_img)
            if mask is None:
                st.error("Could not segment millipede body from background using morphological thresholding.")
            else:
                unrolled, signal, peak_count = classical_engine.extract_skeleton_and_unroll(work_img, mask)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(mask, caption="Segmented Binary Mask", use_container_width=True)
                with c2:
                    st.metric("Estimated Peaks / Sutures", f"{peak_count}")

                if unrolled is not None and signal is not None:
                    st.subheader("Unrolled Body Representation")
                    st.image(unrolled, clamp=True, use_container_width=True, caption="Straightened along centerline axis")

                    st.subheader("1D Transverse Gradient Signal")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(signal, color="#00eb78", linewidth=1.5)
                    ax.set_facecolor("#1e212b")
                    fig.patch.set_facecolor("#1e212b")
                    ax.tick_params(colors="#dcdcdc")
                    ax.grid(True, linestyle="--", alpha=0.3, color="#555")
                    st.pyplot(fig)
else:
    st.info("👆 Please upload an image, choose the sample specimen, or take a webcam snapshot to begin.")
