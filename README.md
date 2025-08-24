# 🔍 DeepFake Detector (Dual-Branch CNN + Handcrafted Features)

This project implements a **DeepFake detection system** using a **dual-branch neural network**:
- One branch is a **CNN (EfficientNet-B0)** for learning visual features.
- The other branch processes **handcrafted features** (color histograms, edges, textures, statistics).
- Both branches are fused for classification into **REAL** or **FAKE**.

The project also includes a **Streamlit web app** where you can:
- Try demo images.
- Upload your own image.
- Get a prediction with confidence score.

---

## 📌 Features
- **Dual-branch model** combining deep CNN and handcrafted features.
- **Two-phase training** (head training + fine-tuning).
- **Early stopping** for better generalization.
- **Streamlit app** for interactive demo.
- Support for both **CPU** and **GPU**.

---

## ⚙️ Tech Stack
- **Python**
- **PyTorch** (EfficientNet-B0 backbone)
- **Torchvision**
- **OpenCV**
- **Scikit-learn & Scikit-image**
- **Streamlit** (UI)
- **NumPy / Pandas / Matplotlib**

---

## 🚀 Getting Started

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧠 Model Training

1. Dataset is expected in `content/face_224/` with metadata in `metadata.csv`.
2. Run training:

   ```bash
   python train.py
   ```
3. The model is saved as `deepfake_detector_dual_branch.pth`.

---

## 🎮 Usage (App Demo)

* Launch the app:

  ```bash
  streamlit run app.py
  ```
* Choose from **4 demo images** or **upload your own image**.
* Click **Predict** → model returns `REAL` or `FAKE` with a confidence score.

---

## 📊 Results

* **EfficientNet-B0 + handcrafted features** improved validation accuracy compared to CNN-only baseline.
* Early stopping prevents overfitting.
* Dual-branch approach is more robust against subtle artifacts.

---