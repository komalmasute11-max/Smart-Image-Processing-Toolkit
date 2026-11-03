import io
import os
import gc

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from rembg import new_session, remove
from diffusers import StableDiffusionPipeline

st.set_page_config(
    page_title="Smart Image Processing & Enhancement Toolkit",
    page_icon="🖼️",
    layout="wide",
)

def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True,
        )


load_css("style.css")

st.markdown(
    """
    <section class="hero-header">
        <div class="hero-logo">🖼️</div>
        <div class="hero-copy">
            <div class="hero-title">Smart Image Processing &amp;<br>
                Enhancement <span>Toolkit</span>
            </div>
            <p>Enhance uploaded images with classic image-processing tools,
            or generate a new image from a text prompt.</p>
        </div>
        <div class="hero-art" aria-hidden="true">✨ 🏞️</div>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_rembg_session():
    return new_session("u2netp")


@st.cache_resource(show_spinner=False)
def load_text_to_image_pipeline():
    """Load Stable Diffusion from the D-drive cache and report clear errors."""
    model_id = os.getenv(
        "TEXT_TO_IMAGE_MODEL",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Local-only prevents downloads while checking the existing D-drive cache.
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir="D:/huggingface-cache/hub",
        local_files_only=True,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    if device == "cuda":
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

    return pipe, device


def load_uploaded_image(uploaded_file):
    """Return a PIL RGB image and an OpenCV BGR image."""
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return pil_image, bgr



def process_image(
    bgr,
    tool,
    brightness=0,
    contrast=1.0,
    denoise_strength=15,
    sharpen_amount=1.0,
    canny_low=100,
    canny_high=200,
):
    if tool == "Original":
        result = bgr.copy()

    elif tool == "Grayscale":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif tool == "Brightness & Contrast":
        result = cv2.convertScaleAbs(
            bgr,
            alpha=float(contrast),
            beta=int(brightness),
        )

    
    
    elif tool == "Denoising":
        strength = max(1, min(10, int(denoise_strength)))

        # Convert to LAB color space to process brightness separately.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply gentle denoising to preserve image details.
        denoised_l = cv2.fastNlMeansDenoising(
            l_channel,
            None,
            h=strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        denoised_lab = cv2.merge(
            [denoised_l, a_channel, b_channel]
        )
        result = cv2.cvtColor(
            denoised_lab,
            cv2.COLOR_LAB2BGR,
        )



    elif tool == "Sharpening":
        amount = float(sharpen_amount)
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2)
        result = cv2.addWeighted(
            bgr, 1.0 + amount, blurred, -amount, 0
        )

    elif tool == "Canny Edge Detection":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            int(canny_low),
            int(canny_high),
        )
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif tool == "Background Removal":
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return remove(
            pil,
            session=get_rembg_session(),
        ).convert("RGBA")

    else:
        result = bgr.copy()

    # Convert BGR to RGB for correct display in Streamlit.
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def show_image_processing():
    st.subheader("Image Processing & Enhancement")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
        key="processing_upload",
    )

    if uploaded_file is None:
        st.info("Upload an image to start using the processing tools.")

        st.markdown(
            """
            <div class="feature-grid">
                <div class="feature-card"><span>⚡</span><b>Fast Processing</b></div>
                <div class="feature-card"><span>🛡️</span><b>High Quality Results</b></div>
                <div class="feature-card"><span>🎨</span><b>Multiple Tools</b></div>
                <div class="feature-card"><span>💎</span><b>Easy to Use</b></div>
            </div>
            <div class="hero-tagline">
                <span></span>
                <em>Turn Your Images Into Something Amazing ✨</em>
                <span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        original_pil, original_bgr = load_uploaded_image(uploaded_file)
    except Exception as exc:
        st.error(f"Could not open this image: {exc}")
        return

    st.sidebar.subheader("Processing settings")
    tool = st.sidebar.selectbox(
        "Choose a tool",
        [
            "Original",
            "Grayscale",
            "Brightness & Contrast",
            "Denoising",
            "Sharpening",
            "Canny Edge Detection",
            "Background Removal",
        ],
    )

    brightness = 0
    contrast = 1.0
    denoise_strength = 10
    sharpen_amount = 1.0
    canny_low = 100
    canny_high = 200

    if tool == "Brightness & Contrast":
        brightness = st.sidebar.slider("Brightness", -100, 100, 0)
        contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
    
    elif tool == "Denoising":
        denoise_strength = st.sidebar.slider(
            "Denoising strength",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
        )

    elif tool == "Sharpening":
        sharpen_amount = st.sidebar.slider(
            "Sharpening amount", 0.1, 2.5, 1.0, 0.1
        )
    elif tool == "Canny Edge Detection":
        canny_low = st.sidebar.slider("Lower threshold", 0, 255, 100)
        canny_high = st.sidebar.slider("Upper threshold", 0, 255, 200)
        if canny_low >= canny_high:
            st.sidebar.warning(
                "Upper threshold should be greater than lower threshold."
            )

    st.caption(f"Selected tool: **{tool}**")

    try:
        processed = process_image(
            original_bgr,
            tool,
            brightness=brightness,
            contrast=contrast,
            denoise_strength=denoise_strength,
            sharpen_amount=sharpen_amount,
            canny_low=canny_low,
            canny_high=canny_high,
        )
    except Exception as exc:
        st.error(f"Image processing failed: {exc}")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Original image**")
        st.image(original_pil, width="stretch")
    with right:
        st.markdown("**Processed image**")
        st.image(processed, width="stretch")

    if isinstance(processed, Image.Image):
        downloadable = image_to_png_bytes(processed)
    else:
        downloadable = image_to_png_bytes(Image.fromarray(processed))

    st.download_button(
        "Download processed image (PNG)",
        data=downloadable,
        file_name="processed_image.png",
        mime="image/png",
    )


def show_text_to_image():
    st.subheader("Text-to-Image Generation")
    st.write(
        "Describe the image you want. The AI model will create a new image from your prompt."
    )

    prompt = st.text_area(
        "Image prompt",
        value="A beautiful red rose in a magical garden, realistic photography",
        height=100,
        help="Be specific about the subject, setting, lighting, and visual style.",
    )
    negative_prompt = st.text_input(
        "Negative prompt (optional)",
        value="blurry, low quality, distorted, text, watermark",
        help="Things you do not want in the generated image.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        width = st.selectbox("Width", [512, 640, 768], index=0)
    with col2:
        height = st.selectbox("Height", [512, 640, 768], index=0)
    with col3:
        steps = st.slider("Inference steps", min_value=10, max_value=35, value=20)

    seed = st.number_input(
        "Seed (use -1 for a random image)",
        min_value=-1,
        max_value=2147483647,
        value=-1,
        step=1,
    )
    generate = st.button(
        "✨ Generate image",
        type="primary",
        width="stretch",
    )

    if not generate:
        st.info(
            "The model is loaded from the local D-drive cache. "
            "CPU generation may take several minutes; use 512 × 512 and 10–15 steps for a quicker test."
        )
        return

    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
        return

    try:
        with st.spinner("Loading the text-to-image model from the local cache..."):
            pipe, device = load_text_to_image_pipeline()

        generator = None
        if int(seed) >= 0:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        with st.spinner(
            f"Generating image on {device.upper()}... Please wait; CPU can be slow."
        ):
            result = pipe(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip() or None,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=7.5,
                generator=generator,
            )
            generated_image = result.images[0]

        st.success("Image generated successfully!")
        st.image(generated_image, caption=prompt.strip(), width="stretch")
        st.download_button(
            "Download generated image (PNG)",
            data=image_to_png_bytes(generated_image),
            file_name="generated_image.png",
            mime="image/png",
        )

    except Exception as exc:
        st.error("Image generation failed. See the details below.")
        st.exception(exc)
        st.markdown(
            """
            **What to check**
            - If a file is missing, the D-drive model cache may be incomplete.
            - If there is an out-of-memory error, close other heavy apps and try 512 × 512 with 10 steps.
            - If a Hugging Face access/terms error appears, follow the model access instructions shown in the error.
            - Do not delete the C-drive cache until the D-drive model loads successfully.
            """
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🖼️</div>
        <div class="sidebar-brand-title">Smart Image<br><span>Toolkit</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a feature",
    ["Image Processing", "Text-to-Image Generation"],
)

if page == "Image Processing":
    show_image_processing()
else:
    show_text_to_image()

st.markdown(
    """
    <div class="app-footer">
        Smart Image Processing &amp; Enhancement Toolkit
        <span>•</span> Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
