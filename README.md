# Brain Tumor MRI Classifier with Integrated Gradients

> A deep learning pipeline that classifies brain MRI scans into four categories and explains its decisions using Integrated Gradients — served through a live web interface.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Results](#results)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Explainability — Integrated Gradients](#explainability--integrated-gradients)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Key Design Decisions](#key-design-decisions)
- [Training Behavior](#training-behavior)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)

---

## Project Overview

This project builds a complete brain tumor classification system from scratch. Given an MRI scan, the model predicts which of four conditions is present: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. Beyond just making a prediction, the system uses **Integrated Gradients** to generate a pixel-level attribution map — a heatmap that highlights which regions of the MRI drove the model's decision. This makes the model's reasoning transparent and interpretable, which is especially important in medical AI.

The full pipeline covers:
1. Data loading and preprocessing from raw MRI images
2. Training a custom CNN from scratch (no pretrained weights)
3. Evaluating with precision, recall, F1, and a confusion matrix
4. Computing Integrated Gradients attributions for correct and misclassified samples
5. Serving predictions and explanations through a Flask web app accessible via a public URL

---

## Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **94.04%** |
| Test Loss | 0.1915 |
| Macro F1 | 0.94 |
| Weighted F1 | 0.94 |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 0.95 | 0.95 | 0.95 |
| Meningioma | 0.93 | 0.86 | 0.89 |
| No Tumor | 0.95 | 0.95 | 0.95 |
| Pituitary | 0.93 | 0.99 | 0.96 |


**Confusion matrix highlights:**
- Pituitary and No Tumor are classified with near-perfect recall
- Meningioma has the lowest recall (84%) — the hardest class, as meningiomas can visually resemble normal tissue at 128×128 resolution
- Zero confusion between No Tumor and any tumor class — clinically the most important boundary

---

## Dataset

**Source:** [Brain Tumor MRI Dataset — Kaggle (Masoud Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Class | Images |
|---|---|
| Glioma | 1,621 |
| Meningioma | 1,775 |
| No Tumor | 2,000 |
| Pituitary | 1,757 |
| **Total** | **7,153** |

**Split:** 70% train / 15% validation / 15% test (stratified, `random_state=42`)

---

## Architecture

### Why a custom CNN instead of a pretrained model?

Transfer learning with EfficientNetB0 (ImageNet weights) was attempted first. It repeatedly crashed mid-epoch on Colab's free GPU due to memory pressure from its 4.4M parameters and 224×224 input requirement. Additionally, a double-normalization bug caused validation loss to explode to 945 — EfficientNetB0 includes an internal Rescaling layer that expects [0, 255] inputs, but the ImageDataGenerator was already dividing by 255. The custom CNN avoids this entire class of problem by placing normalization explicitly as the first layer inside the model.

### Model summary

```
Input: (128, 128, 3) uint8
  Rescaling(1/255)
  Block 1: Conv32 → BN → ReLU → Conv32 → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  Block 2: Conv64 → BN → ReLU → Conv64 → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  Block 3: Conv128 → BN → ReLU → Conv128 → BN → ReLU → MaxPool(2×2) → Dropout(0.30)
  Block 4: Conv256 → BN → ReLU → Conv256 → BN → ReLU → MaxPool(2×2) → Dropout(0.30)
  GlobalAveragePooling2D
  Dense(256, relu, L2=1e-4) → BatchNorm → Dropout(0.50)
  Dense(4, softmax)

Total parameters: 1,241,508 (4.74 MB)
```

### Design decisions explained

**Two Conv layers per block before MaxPool**
Two consecutive 3×3 convolutions have the same receptive field as one 5×5 convolution, but with fewer parameters and an extra nonlinearity — letting the model learn richer features at each spatial scale before resolution is halved.

**Filter doubling: 32 → 64 → 128 → 256**
Each MaxPool halves spatial dimensions. Doubling filters keeps the total information capacity approximately constant through the network. Early layers detect simple patterns (edges, intensity) and need few filters; deep layers distinguish complex class-specific patterns (tumor boundaries, tissue heterogeneity) and need more.

**BatchNormalization after every Conv**
Normalizes activations across the batch before the ReLU gate, preventing internal covariate shift, stabilizing training, and acting as a mild regularizer. Placed before activation following modern CNN convention.

**GlobalAveragePooling instead of Flatten**
After 4 MaxPool layers the feature maps are 8×8. Flattening produces 8×8×256 = 16,384 features — a massive overfitting-prone vector. GAP collapses each of the 256 maps to a single number, yielding a 256-dim vector. Fewer parameters, stronger regularization, and — critically — it preserves spatial correspondence between feature maps and input pixels, which is what makes Integrated Gradients produce clean, interpretable attribution maps.

**Rescaling layer inside the model**
Normalization is applied as the model's first operation, converting uint8 [0,255] to float32 [0,1]. Keeping it inside the model guarantees identical preprocessing at training, evaluation, and web app inference — eliminating the entire class of pipeline bugs encountered with EfficientNetB0.

**Dropout progression: 0.25 → 0.25 → 0.30 → 0.30 → 0.50**
Dropout increases toward the head. Early conv layers extract general features and need light regularization. The dense head has the highest capacity for memorization (65k weights), so 50% dropout there provides the strongest pressure against overfitting.

---

## Explainability — Integrated Gradients

### Why standard gradients are insufficient

Standard gradient attribution computes `∂output/∂input` at the actual image. The problem is **gradient saturation**: once a neuron is strongly activated, its gradient approaches zero even if that pixel clearly matters. Important regions appear artificially dark in the attribution map.

### How Integrated Gradients works

Integrated Gradients (Sundararajan et al., 2017) answers a different question: *"If I slowly fade the image in from a black baseline, how much does each pixel's gradual appearance contribute to the model's output?"*

```
IG(x) = (x − x') × ∫₀¹ ∂F(x' + α(x − x')) / ∂x dα
```

50 interpolated images are generated between baseline (all zeros) and input. Gradients of the softmax probability (not raw logits — more stable) are computed at each interpolated image. The integral is approximated using the **trapezoidal rule**: adjacent gradient pairs are averaged before taking the mean, which is more accurate than a simple mean (Riemann sum).

**SmoothGrad** is applied on top: IG is averaged over 10 slightly noisy copies of the image (σ = 5% of pixel range). Noise cancels across runs; real signal accumulates. This eliminates the speckle artifacts visible in naive IG maps.

### Reading the output

The 3-panel figure shows:
1. **Original MRI** — the raw input
2. **Attribution map** — IG values normalized to [0,1] with `hot` colormap (black→red→yellow→white = low→high attribution)
3. **Overlay** — 50/50 blend of original and attribution map

Bright regions = pixels most responsible for the prediction. In correctly classified tumor cases, these should align with the tumor mass.

---

## Web Application

A Flask backend serves the model and IG computation behind a public URL generated by ngrok. The single-page frontend provides drag-and-drop upload, animated probability bars for all four classes, and the IG overlay rendered inline.

**Request flow:**
1. User uploads MRI via drag-and-drop
2. Frontend POSTs image to `/predict`
3. Backend resizes to 128×128, runs inference, computes IG with SmoothGrad
4. Response includes: label, confidence, all 4 class probabilities, description, and the 3-panel figure as base64 PNG
5. Frontend renders results inline with no page reload

---

## Project Structure

```
brain-tumor-classifier/
├── Brain_Tumor_MRI_Classifier.ipynb    ← complete Colab notebook
│     ├── Section 0-5   : imports, config, preprocessing, dataset, split, generators
│     ├── Section 6-9   : model definition, compile, callbacks, training
│     ├── Section 10-12 : evaluation, confusion matrix, training curve
│     ├── Section 13    : Integrated Gradients (compute + visualize)
│     └── Section 14    : Flask web app + ngrok deployment
└── README.md
```

---

## How to Run

### Prerequisites
- Google Colab with GPU runtime (`Runtime → Change runtime type → T4 GPU`)
- Google Drive with dataset at `MyDrive/Personal Projects/Brain Tumor Dataset/brain-tumor-mri-dataset/`
- Free ngrok account at [dashboard.ngrok.com](https://dashboard.ngrok.com) (web app only)

### Steps

**1. Train the model (Sections 0–12)**
Run cells sequentially. Training takes ~25 min on T4 GPU. Best weights saved automatically to Drive as `best_custom_cnn.keras`.

**2. Save the test set (run once after training)**
```python
np.save('.../X_test.npy', X_test)
np.save('.../y_test.npy', y_test)
```
This guarantees reproducible evaluation in all future sessions.

**3. Run IG analysis without retraining (Section 13)**
```python
model = tf.keras.models.load_model(MODEL_SAVE)
X_test = np.load('.../X_test.npy')
y_test = np.load('.../y_test.npy')
# Then run IG cells normally
```

**4. Launch web app (Section 14)**
```python
!pip install -q flask flask-cors pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
url = run_flask_app(model, CLASS_NAMES, CLASS_INFO, IMG_SIZE)
```

---

## Key Design Decisions

| Decision | Alternative | Why |
|---|---|---|
| Custom CNN | EfficientNetB0 | Pretrained model crashed on Colab; double-normalization bug caused val_loss=945; custom model is 3.5× smaller |
| 128×128 input | 224×224 | 4× less memory; enables stable Colab training; features visible at this resolution |
| GlobalAveragePooling | Flatten | 64× fewer head parameters; spatial correspondence for clean IG maps |
| Rescaling inside model | Generator rescale | Normalization identical at train/eval/inference; eliminates pipeline bugs |
| Trapezoidal IG | Riemann sum | More accurate integral with same computation cost |
| SmoothGrad | Raw IG | Eliminates speckle noise; cleaner spatial attribution |
| Save X_test.npy | Rebuild split each session | Prevents label-image mismatch from non-deterministic `os.listdir()` ordering |

---

## Training Behavior

The training curve shows epoch-to-epoch fluctuation in validation accuracy, particularly between epochs 15–30. This is a **Colab artifact**: Colab's free tier occasionally interrupts mid-epoch to reclaim GPU resources. When this happens, Keras evaluates on fewer gradient updates, producing noisy readings. The `self._interrupted_warning()` messages in the training log confirm this.

The underlying trend is clean: val_accuracy climbs from ~25% to ~94% over 40 epochs. EarlyStopping restores the best weights (epoch 36). The test accuracy of 94.3% on the held-out set confirms genuine generalization.

---

## Limitations

**Skull/boundary attribution in IG maps**
Some IG maps highlight skull edges rather than tumor tissue exclusively. This indicates the model partially learned MRI scan orientation (axial/sagittal/coronal) as a class signal — different classes tend to appear in different orientations in this dataset.

**Meningioma recall (84%)**
Meningiomas grow on the outer membrane and can be visually subtle at 128×128. 42 out of 267 were misclassified. Higher resolution or a pretrained backbone may help.

**Colab session limit**
The web app runs from a Colab runtime (12-hour limit, no persistent URL). Production deployment would require containerization.

**Dataset size**
7,153 images is small by medical AI standards. Clinical deployment would require significantly more data and multi-site validation.

---

## Future Improvements

**Model**
- Attention mechanisms (SE blocks, CBAM) to explicitly focus on relevant spatial regions
- Higher resolution (224×224) with MobileNetV3 or EfficientNetV2-S for meningioma recall
- Test-Time Augmentation (TTA) for 1–2% accuracy gain at no training cost

**Explainability**
- GradCAM comparison to validate IG maps against an independent method
- Quantitative faithfulness evaluation using AOPC pixel perturbation scores
- Blurred image baseline for IG to reduce skull/border attribution artifacts

**Deployment**
- Docker containerization for cloud deployment (AWS, GCP, Azure)
- FastAPI migration for async, concurrent request handling
- Persistent deployment with model versioning

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning | TensorFlow / Keras 2.20 |
| Explainability | Custom IG implementation + SmoothGrad |
| Image processing | OpenCV, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web backend | Flask + flask-cors |
| Tunneling | pyngrok |
| Training environment | Google Colab (T4 GPU) |
| Language | Python 3.12 |

---

*For educational and research purposes only. Not a medical diagnostic tool.*
