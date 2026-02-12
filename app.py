import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Pathology Assistant", layout="wide")
# -----------------------------
# PREMIUM UI STYLE
# -----------------------------
st.markdown("""
<style>
.report-box {
    background-color: #f8f9fc;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #4a90e2;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.metric-card {
    background-color: #f9fbff;
    color: #111111;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e6ecf5;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.06);
    text-align: center;
    font-size: 18px;
    font-weight: 600;
}


.risk-high {
    color: #ff2b2b;
    font-weight: bold;
    font-size: 20px;
}

.risk-medium {
    color: #ff9800;
    font-weight: bold;
    font-size: 20px;
}

.risk-low {
    color: #00c853;
    font-weight: bold;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# CLASS LABELS
# -----------------------------
cell_classes = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate"
]

tissue_classes = ["Normal Tissue", "Tumor Tissue"]

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD CLASSIFICATION MODELS
# -----------------------------
@st.cache_resource
def load_models():
    cell_model = resnet18(weights=None)
    cell_model.fc = nn.Linear(cell_model.fc.in_features, 5)
    cell_model.load_state_dict(torch.load("sipakmed_model.pth", map_location=device))
    cell_model.to(device)
    cell_model.eval()

    tissue_model = resnet18(weights=None)
    tissue_model.fc = nn.Linear(tissue_model.fc.in_features, 2)
    tissue_model.load_state_dict(torch.load("pcam_model.pth", map_location=device))
    tissue_model.to(device)
    tissue_model.eval()

    return cell_model, tissue_model

cell_model, tissue_model = load_models()

# -----------------------------
# SEGMENTATION MODEL (U-NET)
# -----------------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU()
        )

        self.up = nn.ConvTranspose2d(64,32,2,stride=2)

        self.dec = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU()
        )

        self.out = nn.Conv2d(32,1,1)

    def forward(self,x):
        e1 = self.enc1(x)
        p = self.pool(e1)
        e2 = self.enc2(p)

        up = self.up(e2)
        cat = torch.cat([up,e1],dim=1)

        d = self.dec(cat)
        return torch.sigmoid(self.out(d))

@st.cache_resource
def load_seg_model():
    model = UNet().to(device)
    model.load_state_dict(torch.load("sicap_segmentation.pth", map_location=device), strict=False)
    model.eval()
    return model

seg_model = load_seg_model()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# MEDICAL IMAGE VALIDATION
# -----------------------------
def is_medical_image(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    avg_hue = hsv[:,:,0].mean()
    return 110 < avg_hue < 170

# -----------------------------
# GRAD-CAM
# -----------------------------
def generate_heatmap(model, image):
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    target_layer = model.layer4[1].conv2
    target_layer.register_forward_hook(save_activation)
    target_layer.register_full_backward_hook(save_gradient)

    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    confidence = probs.max().item()
    pred_class = probs.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    original = cv2.resize(np.array(image), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + original

    return pred_class, confidence, original, heatmap, overlay

# -----------------------------
# MORPHOLOGY ENGINE
# -----------------------------
def analyze_morphology(image):
    img = cv2.resize(np.array(image), (224,224))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold for dark nuclei
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 20]
    total_area = sum(areas)

    nucleus_ratio = total_area / (224*224)

    density_score = len(areas) / 50
    density_score = min(density_score, 1)

    irregularity_scores = []
    for c in contours[:15]:
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if area > 0:
            irregularity_scores.append(perimeter / (area**0.5))

    irregularity = np.mean(irregularity_scores) if irregularity_scores else 0

    darkness = 1 - (gray.mean() / 255)

    # TEXT INTERPRETATION
    indicators = []

    if nucleus_ratio > 0.25:
        indicators.append("Large nucleus regions detected")

    if density_score > 0.5:
        indicators.append("High cell/tissue density")

    if irregularity > 5:
        indicators.append("Irregular structural boundaries")

    if darkness > 0.4:
        indicators.append("Dark stain concentration observed")

    return {
        "indicators": indicators,
        "nucleus_ratio": nucleus_ratio,
        "density": density_score,
        "irregularity": irregularity,
        "darkness": darkness
    }

# -----------------------------
# PATCH SCANNER
# -----------------------------
def scan_patches(image):
    img = cv2.resize(np.array(image), (224,224))
    h, w, _ = img.shape

    patch_size = 56
    suspicious_regions = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]

            patch_resized = cv2.resize(patch, (224,224))
            tensor = torch.tensor(patch_resized/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

            # Auto decide model
            if patch.mean() < 150:
                model = tissue_model
                classes = tissue_classes
            else:
                model = cell_model
                classes = cell_classes

            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                conf = probs.max().item()
                pred = probs.argmax(dim=1).item()

            # If suspicious
            if conf > 0.7:
                suspicious_regions.append((x, y, patch_size, patch_size))

    return img, suspicious_regions

def generate_clinical_report(label, confidence, morph):
    density = morph["density"]
    irregularity = morph["irregularity"]
    darkness = morph["darkness"]

    # Risk logic
    if confidence > 0.85 and density > 0.5:
        risk = "HIGH"
        risk_class = "risk-high"
    elif confidence > 0.70:
        risk = "MODERATE"
        risk_class = "risk-medium"
    else:
        risk = "LOW"
        risk_class = "risk-low"

    findings = []

    if density > 0.5:
        findings.append("High cellular density detected")

    if irregularity > 5:
        findings.append("Irregular structural morphology observed")

    if darkness > 0.4:
        findings.append("Dark nuclear staining pattern present")

    if confidence > 0.9:
        findings.append("Strong AI diagnostic confidence")

    return risk, risk_class, findings

# -----------------------------
# UI TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧫 Cell Analysis",
    "🧬 Tissue Analysis",
    "🤖 Auto Detect",
    "🧠 Tumor Region",
    "ℹ️ LIVE PATCH SCANNER"
])

# -----------------------------
# TAB 1 — CELL
# -----------------------------
with tab1:
    st.header("Cell Classification (SIPaKMeD)")
    file = st.file_uploader("Upload Cell Image", type=["bmp","jpg","png"])

    if file:
        image = Image.open(file).convert("RGB")

        if not is_medical_image(image):
            st.error("⚠️ Not a valid microscopy image")
            st.stop()

        pred, conf, original, heatmap, overlay = generate_heatmap(cell_model, image)
        morph = analyze_morphology(image)

        st.subheader(f"Prediction: {cell_classes[pred]}")
        st.write(f"Confidence: {conf*100:.2f}%")

        # IMAGE ROW
        col1, col2, col3 = st.columns(3)
        col1.image(original, caption="Original")
        col2.image(heatmap, caption="Heatmap")
        col3.image(overlay.astype(np.uint8), caption="Overlay")

        # MORPHOLOGY TEXT
        st.subheader("🔬 Morphological Analysis")

        for text in morph["indicators"]:
            st.write("•", text)

        st.markdown("**Metrics:**")
        st.write(f"Nucleus Area Ratio: {morph['nucleus_ratio']*100:.2f}%")
        st.write(f"Density Score: {morph['density']:.2f}")
        st.write(f"Irregularity Index: {morph['irregularity']:.2f}")
        st.write(f"Darkness Level: {morph['darkness']:.2f}")

        # CLINICAL REPORT
        risk, risk_class, findings = generate_clinical_report(cell_classes[pred], conf, morph)

        st.markdown("### 🧾 AI Clinical Interpretation")

        report_html = f"""
        <div class="report-box">
        <b>Diagnosis:</b> {cell_classes[pred]} <br>
        <b>Confidence:</b> {conf*100:.2f}% <br>
        <b>Risk Level:</b> <span class="{risk_class}">{risk}</span>
        <br><br>
        <b>AI Findings:</b>
        <ul>
        {''.join([f'<li>{f}</li>' for f in findings])}
        </ul>
        </div>
        """

        st.markdown(report_html, unsafe_allow_html=True)

        # -----------------------------
        # 🎯 RISK GAUGE METER
        # -----------------------------
        st.markdown("### 🎯 AI Risk Gauge")

        if risk == "HIGH":
            risk_percent = 90
            color = "#ff2b2b"
        elif risk == "MODERATE":
            risk_percent = 70
            color = "#ff9800"
        else:
            risk_percent = 40
            color = "#00c853"


        # Color logic
        if risk_percent >= 85:
            color = "#ff2b2b"   # Red
            label = "HIGH RISK"
        elif risk_percent >= 70:
            color = "#ff9800"   # Orange
            label = "MODERATE RISK"
        else:
            color = "#00c853"   # Green
            label = "LOW RISK"

        gauge_html = f"""
        <div style="text-align:center;">
            <div style="
                width:140px;
                height:140px;
                border-radius:50%;
                border:12px solid {color};
                display:flex;
                align-items:center;
                justify-content:center;
                margin:auto;
                font-size:28px;
                font-weight:bold;
                color:{color};
                background:#ffffff;
                box-shadow:0 0 12px rgba(0,0,0,0.15);
            ">
                {risk_percent}%
            </div>
            <div style="margin-top:10px; font-size:18px; font-weight:600; color:{color};">
                {label}
            </div>
        </div>
        """

        st.markdown(gauge_html, unsafe_allow_html=True)


        # DASHBOARD
        st.markdown("### 📊 Morphology Dashboard")

        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(f"<div class='metric-card'><b>Nucleus Area</b><br>{morph['nucleus_ratio']*100:.1f}%</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><b>Density</b><br>{morph['density']:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><b>Irregularity</b><br>{morph['irregularity']:.2f}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><b>Darkness</b><br>{morph['darkness']:.2f}</div>", unsafe_allow_html=True)


# -----------------------------
# TAB 2 — TISSUE
# -----------------------------
with tab2:
    st.header("Tissue Tumor Detection (PCam)")
    file = st.file_uploader("Upload Tissue Image", type=["tif","jpg","png"])

    if file:
        image = Image.open(file).convert("RGB")

        if not is_medical_image(image):
            st.error("⚠️ Not a valid pathology tissue image")
            st.stop()

        pred, conf, original, heatmap, overlay = generate_heatmap(tissue_model, image)

        st.subheader(f"Prediction: {tissue_classes[pred]}")
        st.write(f"Confidence: {conf*100:.2f}%")

        col1, col2, col3 = st.columns(3)
        col1.image(original, caption="Original")
        col2.image(heatmap, caption="Heatmap")
        morph = analyze_morphology(image)

        st.subheader("🔬 Morphological Analysis")

        for text in morph["indicators"]:
            st.write("•", text)

        st.markdown("**Metrics:**")
        st.write(f"Region Density: {morph['density']:.2f}")
        st.write(f"Structural Irregularity: {morph['irregularity']:.2f}")
        st.write(f"Dark Cluster Ratio: {morph['darkness']:.2f}")

        col3.image(overlay.astype(np.uint8), caption="Overlay")

# -----------------------------
# TAB 3 — AUTO
# -----------------------------
with tab3:
    st.header("Automatic Image Type Detection")
    file = st.file_uploader("Upload Any Image", type=["bmp","jpg","png","tif"])

    if file:
        image = Image.open(file).convert("RGB")

        if file.name.endswith(".tif"):
            pred, conf, original, heatmap, overlay = generate_heatmap(tissue_model, image)
            label = tissue_classes[pred]
        else:
            pred, conf, original, heatmap, overlay = generate_heatmap(cell_model, image)
            label = cell_classes[pred]

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {conf*100:.2f}%")

        col1, col2, col3 = st.columns(3)
        col1.image(original)
        col2.image(heatmap)
        col3.image(overlay.astype(np.uint8))

# -----------------------------
# TAB 4 — SEGMENTATION
# -----------------------------
with tab4:
    st.header("Tumor Region Segmentation (SICAPv2)")

    file = st.file_uploader("Upload Tissue Image", type=["jpg","png","tif"])

    if file:
        image = Image.open(file).convert("RGB")
        img = cv2.resize(np.array(image), (224,224))

        img_tensor = torch.tensor(img/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask = seg_model(img_tensor)[0][0].cpu().numpy()

        # 🔥 LOWER THRESHOLD FOR BETTER REGION VISIBILITY
        mask = (pred_mask > 0.35).astype(np.uint8)*255


        red_mask = np.zeros_like(img)
        red_mask[:,:,0] = mask

        overlay = cv2.addWeighted(img, 0.7, red_mask, 0.6, 0)

        col1, col2, col3 = st.columns(3)
        col1.image(img, caption="Original")
        col2.image(mask, caption="Predicted Mask")
        col3.image(overlay, caption="Tumor Overlay")
        tumor_percent = (mask > 0).sum() / (224*224)

        st.subheader("🧠 Tumor Morphology Report")

        st.write(f"Tumor Area: {tumor_percent*100:.2f}%")

        if tumor_percent > 0.30:
            st.write("• Large tumor region involvement")

        elif tumor_percent > 0.10:
            st.write("• Moderate tumor presence")

        else:
            st.write("• Small localized tumor region")


# -----------------------------
# TAB 5 — LIVE PATCH SCANNER
# -----------------------------
with tab5:
    st.header("🔍 Live Patch AI Scanner (Localized Detection)")

    file = st.file_uploader("Upload Image for Region Scan", type=["bmp","jpg","png","tif"])

    if file:
        image = Image.open(file).convert("RGB")

        st.image(image, caption="Uploaded Image", width=300)

        img, regions = scan_patches(image)

        # Draw rectangles
        for (x,y,w,h) in regions:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        st.subheader("Detected Suspicious Regions")
        st.image(img, caption="Red boxes = AI-detected regions")

        st.write(f"Regions detected: {len(regions)}")

        if len(regions) > 6:
            st.warning("High abnormality distribution detected")

        elif len(regions) > 2:
            st.info("Moderate suspicious activity")

        else:
            st.success("Mostly normal structure")

