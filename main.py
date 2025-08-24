import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from skimage.feature import local_binary_pattern
from PIL import Image
import os

# --- Handcrafted feature extractor ---
def extract_features(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    hist = cv2.calcHist([image], [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    color_hist = hist.flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
    cv2.normalize(edge_hist, edge_hist)
    edge_hist = edge_hist.flatten()

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    stat_features = [np.mean(gray), np.std(gray), np.median(gray),
                     np.percentile(gray, 25), np.percentile(gray, 75)]
    h = gray.shape[0]
    regions = [gray[:h//3], gray[h//3:2*h//3], gray[2*h//3:]]
    for region in regions:
        stat_features.extend([np.mean(region), np.std(region), np.median(region),
                              np.percentile(region, 25), np.percentile(region, 75)])
    stat_features = np.array(stat_features, dtype=np.float32)

    return np.concatenate([color_hist, edge_hist, lbp_hist, stat_features])

# --- Model Definition ---
class DualBranchEfficientNet(nn.Module):
    def __init__(self, feature_dim=262, pretrained=False):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.cnn = backbone.features
        num_cnn_features = backbone.classifier[1].in_features  # 1280

        self.img_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(num_cnn_features),
            nn.Linear(num_cnn_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.feat_fc = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, features):
        x = self.cnn(image)
        x = self.img_fc(x)
        y = self.feat_fc(features)
        combined = torch.cat((x, y), dim=1)
        out = self.classifier(combined)
        return out

# --- Load model ---
MODEL_PATH = "deepfake_detector_dual_branch.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
model = DualBranchEfficientNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Prediction function ---
def predict(image: np.ndarray):
    IMG_SIZE = 224
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    handcrafted_features = extract_features(img_resized)
    handcrafted_features = torch.from_numpy(handcrafted_features.astype(np.float32)).unsqueeze(0).to(device)

    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_norm, handcrafted_features)
        prob = output.item()
        label = "FAKE" if prob >= 0.5 else "REAL"
    return label, prob

# --- Streamlit App ---
st.title("🔍 DeepFake Detector")

if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

# --- Demo images ---
demo_images = [
    "content/rubaqwhhzw.jpg",
    "content/kyfwctavtq.jpg",
    "content/ryryihcyjn.jpg",
    "content/sjuztahktm.jpg",
]

st.subheader("Choose a demo image or upload your own")
cols = st.columns(4)

for i, img_path in enumerate(demo_images):
    if os.path.exists(img_path):
        img = Image.open(img_path)
        if cols[i].button(f"Select Demo {i+1}"):
            st.session_state["selected_image"] = np.array(img)
            st.session_state["prediction"] = None  # reset old prediction
        cols[i].image(img, use_container_width=True)
    else:
        cols[i].warning(f"Missing demo{i+1}")

# --- Upload image ---
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.session_state["selected_image"] = np.array(Image.open(uploaded_file))
    st.session_state["prediction"] = None  # reset old prediction

# --- Show selected image ---
if st.session_state["selected_image"] is not None:
    st.image(st.session_state["selected_image"], caption="Selected Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            label, prob = predict(st.session_state["selected_image"])  # <-- your function
            st.session_state["prediction"] = (label, prob)

# --- Show prediction ---
if st.session_state["prediction"] is not None:
    label, prob = st.session_state["prediction"]
    if label=="REAL":
        st.success(f"Prediction: **{label}** (Confidence: {prob:.4f})")
    else:
        st.error(f"Prediction: **{label}** (Confidence: {prob:.4f})")

