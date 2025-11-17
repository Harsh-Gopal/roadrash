# Home.py  -- fixed for Streamlit versions that require use_column_width
import streamlit as st
from pathlib import Path
import base64
from PIL import Image, UnidentifiedImageError
import io
import traceback

# ---------- Page config ----------
st.set_page_config(
    page_title="Automated Road Damage Detection — Research Demo",
    page_icon="🛣️",
    layout="wide",
)

# ---------- Paths (update if necessary) ----------
PDF_PATH = "/Users/harshgopal/yolo_test/RoadDamageDetection/resource/roadrashpaper.pdf"
BANNER_PATH = "./resource/my_banner.png"  # path to your banner image (optional)

# ---------- Helpers ----------
def embed_pdf(pdf_path: str, height: int = 420):
    p = Path(pdf_path)
    if not p.exists():
        return None
    pdf_bytes = p.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" type="application/pdf"></iframe>'
    return html, pdf_bytes

def show_banner(path: str):
    p = Path(path)
    if not p.exists():
        st.markdown("<h1 style='margin-bottom:0.1rem'>Automated Road Damage Detection System</h1>", unsafe_allow_html=True)
        st.markdown("**Drone + Vehicle Cameras • Deep Learning (YOLOv8 / Mask R-CNN / EfficientDet)**")
        return
    try:
        with Image.open(p) as img:
            max_width = 2200
            if img.width > max_width:
                ratio = max_width / float(img.width)
                new_h = int(img.height * ratio)
                img = img.resize((max_width, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            # use_column_width works on more Streamlit versions
            st.image(buf.read(), use_column_width=True)
    except UnidentifiedImageError:
        st.warning("Banner image file exists but is not a valid image. Showing text header instead.")
        st.markdown("<h1 style='margin-bottom:0.1rem'>Automated Road Damage Detection System</h1>", unsafe_allow_html=True)
        st.markdown("**Drone + Vehicle Cameras • Deep Learning (YOLOv8 / Mask R-CNN / EfficientDet)**")
    except Exception:
        st.error("Failed to load banner image — showing text header. See trace below.")
        st.markdown("<h1 style='margin-bottom:0.1rem'>Automated Road Damage Detection System</h1>", unsafe_allow_html=True)
        st.markdown("**Drone + Vehicle Cameras • Deep Learning (YOLOv8 / Mask R-CNN / EfficientDet)**")
        st.text(traceback.format_exc())

# ---------- Header / banner ----------
show_banner(BANNER_PATH)
st.divider()

# ---------- Title ----------
st.title("Automated Road Damage Detection — Research Demo")
st.subheader("Drone + Vehicle-Mounted Cameras  •  Deep Learning (YOLOv8 / Mask R-CNN / EfficientDet)")

st.divider()

# ---------- Main content: left (text) and right (paper) ----------
left, right = st.columns([2, 1])

with left:
    st.markdown("### Abstract (short)")
    st.markdown(
        """
        This research presents an automated road damage detection system combining drone (UAV) imagery and vehicle-mounted camera footage.
        The pipeline uses deep learning models (YOLOv8 for real-time detection, Mask R-CNN for segmentation and EfficientDet for edge deployment), applies transfer learning and data augmentation on mixed datasets (RDD2020/2022 + custom captures), and outputs geo-tagged detections with severity estimates to enable proactive road maintenance.
        """
    )

    st.markdown("### Key research objectives")
    st.markdown(
        """
        - Integrate aerial (drone) and ground (vehicle) imaging for robust coverage.  
        - Use transfer learning (YOLOv8 / Mask R-CNN) and fine-tuning on combined datasets.  
        - Produce real-time detection on vehicle streams and high-accuracy segmentation from drone imagery.  
        - Provide severity scoring and geo-tagged visualization for maintenance planning.
        """
    )

    st.markdown("### Models & approach")
    st.markdown(
        """
        - **YOLOv8**: fast, real-time multi-class detection (pothole, longitudinal crack, transverse crack, alligator crack).  
        - **Mask R-CNN**: pixel-level segmentation useful for severity & area computation.  
        - **EfficientDet**: lightweight option for edge deployment.  
        - **Data**: RDD2020 / RDD2022 + custom drone/vehicle captures.  
        - **Preprocessing**: tiling for aerial imagery, brightness/scale augmentation, annotation in YOLO format for detection and masks for segmentation.
        """
    )

    st.markdown("### System Overview")
    st.markdown(
        """
        The system consists of the following layers:
        1. Data Acquisition (drone & vehicle cameras)  
        2. Preprocessing (tiling, normalization, augmentation)  
        3. Detection & Segmentation (YOLOv8 for detection, Mask R-CNN for segmentation)  
        4. Post-processing (duplicate removal, severity scoring, geo-tagging)  
        5. Visualization & Reporting (dashboards, maps)
        """
    )

    st.markdown("### Experimental Setup (summary)")
    st.markdown(
        """
        - Hardware: Training on GPU (RTX series recommended). Inference/experiments can run on Apple M-series using PyTorch MPS for small tests.  
        - Training: Transfer learning with YOLOv8, 50–100 epochs depending on dataset size; batch size tuned to GPU memory.  
        - Evaluation: Use mAP, IoU, Precision, Recall, F1, and FPS for speed.
        """
    )

with right:
    st.markdown("### Research paper (preview)")
    pdf_embed = embed_pdf(PDF_PATH, height=560)
    if pdf_embed:
        html, pdf_bytes = pdf_embed
        st.markdown(html, unsafe_allow_html=True)
        # single unique download button (key to avoid duplicates)
        st.download_button(
            "Download research paper (PDF)",
            data=pdf_bytes,
            file_name="roadrashpaper.pdf",
            mime="application/pdf",
            key="download_research_paper_home",
        )
    else:
        st.warning(f"Research PDF not found at `{PDF_PATH}`. Please check the path in Home.py.")

st.divider()

# ---------- Sidebar: minimal project info (NO demo controls) ----------
st.sidebar.title("Project")
st.sidebar.markdown("Automated Road Damage Detection")
st.sidebar.markdown("- Drone + Vehicle cameras")
st.sidebar.markdown("- YOLOv8 / Mask R-CNN")
st.sidebar.markdown("---")
st.sidebar.header("Contact")
st.sidebar.markdown("- Harsh Gopal")
st.sidebar.markdown("- Email: your.email@domain.com")
st.sidebar.markdown("- GitHub: (replace with your repo link)")

st.divider()
st.markdown("**Notes:** This home page contains project description and the embedded research paper. Use the sidebar navigation (left) for different detection demos.")
