from io import BytesIO

import streamlit as st
from PIL import Image

from providers import PROVIDERS


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MILLICOUNT",
    page_icon="🐛",
    layout="wide"
)


# ============================================================
# API KEY RESOLUTION (generic across providers)
# ============================================================

def resolve_api_key(provider_key: str) -> str | None:
    """
    Look for a given provider's API key, in order of priority:
      1. The current Streamlit session (user typed it in the sidebar)
      2. st.secrets (.streamlit/secrets.toml)
      3. Environment variables
    Returns None if nothing is found instead of raising, so the app
    can still render and prompt the user for a key.
    """

    provider = PROVIDERS[provider_key]
    session_key = f"api_key__{provider_key}"

    if st.session_state.get(session_key):
        return st.session_state[session_key]

    # st.secrets raises if no secrets.toml exists at all, so guard it.
    try:
        secret_key = st.secrets.get(provider.secrets_key)
        if secret_key:
            return secret_key
    except Exception:
        pass

    import os

    value = os.environ.get(provider.secrets_key)
    if value:
        return value

    if provider.env_fallback:
        return os.environ.get(provider.env_fallback)

    return None


# ============================================================
# HEADER
# ============================================================

st.title("🐛 MILLICOUNT")
st.caption("AI-powered millipede segment and leg estimation")

st.info(
    "Upload an image or capture a frame, pick an AI provider, "
    "and get a segment/leg count estimate."
)


# ============================================================
# SIDEBAR: PROVIDER + API KEY
# ============================================================

with st.sidebar:
    st.subheader("⚙️ Settings")

    provider_key = st.selectbox(
        "AI provider",
        options=list(PROVIDERS.keys()),
        format_func=lambda k: PROVIDERS[k].label,
    )
    provider = PROVIDERS[provider_key]

    model = st.text_input(
        "Model",
        value=st.session_state.get(f"model__{provider_key}", provider.default_model),
        help="Override with any vision-capable model your key has access to.",
    )
    st.session_state[f"model__{provider_key}"] = model

    api_key = resolve_api_key(provider_key)

    if api_key:
        st.success(f"{provider.label} API key loaded.")
        if st.button("Use a different key"):
            st.session_state.pop(f"api_key__{provider_key}", None)
            st.rerun()
    else:
        st.warning(f"No {provider.label} API key found.")
        typed_key = st.text_input(
            f"Enter your {provider.label} API key",
            type="password",
            help=(
                f"Get a key at {provider.signup_url}. You can also set this "
                f"permanently via `.streamlit/secrets.toml` "
                f"(`{provider.secrets_key}`) or an environment variable."
            ),
        )
        if typed_key:
            st.session_state[f"api_key__{provider_key}"] = typed_key
            st.rerun()

api_key = resolve_api_key(provider_key)

if not api_key:
    st.stop()


# ============================================================
# INPUT MODE
# ============================================================

mode = st.radio(
    "Input Mode",
    ["📁 Upload Image", "📷 Camera"],
    horizontal=True,
)

image = None


# ============================================================
# UPLOAD MODE
# ============================================================

if mode == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a millipede image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Could not read this image.")


# ============================================================
# CAMERA MODE
# ============================================================

else:

    camera_image = st.camera_input("Take a picture of the millipede")

    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert("RGB")
        except Exception:
            st.error("Could not read the camera image.")


# ============================================================
# DISPLAY IMAGE
# ============================================================

if image is not None:

    st.divider()

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Analysis")

        analyze_button = st.button(
            f"🔬 Analyze with {provider.label}",
            type="primary",
            use_container_width=True,
        )

        if analyze_button:

            with st.spinner(f"{provider.label} is examining the millipede..."):

                try:
                    buffer = BytesIO()
                    image.convert("RGB").save(buffer, format="JPEG", quality=95)
                    image_bytes = buffer.getvalue()

                    result = provider.analyze(image_bytes, api_key, model)

                    # ----------------------------------------
                    # Millipede detection
                    # ----------------------------------------

                    if not result.millipede_detected:
                        st.error(
                            "No millipede was confidently "
                            "identified in this image."
                        )

                    else:

                        # ------------------------------------
                        # Main results
                        # ------------------------------------

                        m1, m2 = st.columns(2)
                        m1.metric("🐛 Body Segments", result.segment_count)
                        m2.metric("🦵 Estimated Legs", result.leg_count)

                        # ------------------------------------
                        # Confidence
                        # ------------------------------------

                        confidence_percentage = result.confidence * 100

                        st.progress(min(max(result.confidence, 0.0), 1.0))
                        st.write(f"Confidence: **{confidence_percentage:.0f}%**")

                        # ------------------------------------
                        # Visible segments
                        # ------------------------------------

                        st.write(
                            f"Clearly visible segments: "
                            f"**{result.visible_segments}**"
                        )

                        # ------------------------------------
                        # Explanation
                        # ------------------------------------

                        st.subheader("🤖 AI Explanation")
                        st.write(result.explanation)

                        # ------------------------------------
                        # Warning
                        # ------------------------------------

                        if result.confidence < 0.70:
                            st.warning(
                                "The image is ambiguous, so "
                                "the reported count should be "
                                "treated as an estimate."
                            )
                        else:
                            st.success("Millipede successfully analyzed.")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"MILLICOUNT is currently using {provider.label} for visual estimation. "
    "Results are AI-generated and may contain counting errors."
)
