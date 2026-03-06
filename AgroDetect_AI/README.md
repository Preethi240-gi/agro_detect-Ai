# 🌿 AgroDetect AI – Plant Disease Classification Engine

> AI + Transfer Learning (MobileNetV2) to classify crop diseases from leaf images.
> Stack: Python · TensorFlow · Keras · CNN · OpenCV

---

## 📁 Project Structure

```
AgroDetect_AI/
├── data/
│   ├── raw/           ← downloaded / extracted dataset
│   └── processed/     ← resized 224×224 images (auto-created)
├── models/            ← saved .h5 model (auto-created)
├── notebooks/
│   └── AgroDetect_Walkthrough.ipynb
├── outputs/           ← charts, reports, labels (auto-created)
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── tests/
└── requirements.txt
```

---

## ⚡ Quick-Start — Every Command in Order

Open a **VS Code Terminal** (`Ctrl + `` ` ``) and run the commands below.

---

### ✅ Step 0 – Set Up Environment

```bash
# 1. Create & activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install all dependencies
pip install -r requirements.txt
```

---

### ✅ Step 1 – Download & Prepare Dataset

**Option A – Kaggle CLI** *(recommended)*
```bash
# Install Kaggle
pip install kaggle

# Place your kaggle.json in ~/.kaggle/  then run:
python src/prepare_data.py --kaggle
```

**Option B – Local ZIP you already have**
```bash
python src/prepare_data.py --zip path/to/plantvillage.zip
```

**Expected output:**
```
[Kaggle] Downloading PlantVillage dataset...
[Preprocess] Resizing images to (224, 224)...
100%|████████| 54306/54306 [02:14<00:00]
──────────────────────────────────────────────────
  Apple___Apple_scab                          3170 images
  Apple___Black_rot                           2764 images
  ...
  TOTAL                                      54306 images
──────────────────────────────────────────────────
```

---

### ✅ Step 2 – Train the Model

```bash
python src/train.py
```

**Expected output:**
```
[1/5] Loading data generators...
    Classes found: 38
[2/5] Building MobileNetV2 transfer learning model...
    Total params: 2,422,310
    Trainable params: 527,654
[3/5] Setting up callbacks...
[4/5] Training model...
Epoch 1/20
1357/1357 [==============================] - 94s 68ms/step
    loss: 1.2943 - accuracy: 0.6721 - val_accuracy: 0.8834
...
Epoch 15/20  (EarlyStopping may kick in earlier)
✅ Training complete!
   Model saved  → models/agrodetect_mobilenet.h5
   Curves saved → outputs/training_curves.png
```

---

### ✅ Step 3 – Predict on a Leaf Image

```bash
python src/predict.py --image data/raw/sample_leaf.jpg
```

With options:
```bash
python src/predict.py \
  --image  data/raw/sample_leaf.jpg \
  --model  models/agrodetect_mobilenet.h5 \
  --labels outputs/class_labels.json \
  --top    5
```

**Expected output:**
```
=======================================================
  🌿 AgroDetect AI – Prediction Results
=======================================================
  #1  Tomato___Late_blight                    94.7%  ██████████████████████████████
  #2  Tomato___Early_blight                    3.1%  █
  #3  Tomato___Leaf_Mold                       1.2%
  #4  Potato___Late_blight                     0.6%
  #5  Tomato___Bacterial_spot                  0.4%
=======================================================

✅ Top Prediction: Tomato___Late_blight  (94.7% confidence)
📊 Result chart saved → outputs/prediction_result.png
```

---

### ✅ Step 4 – Evaluate Model Performance

```bash
python src/evaluate.py
```

**Expected output:**
```
[1/4] Loading model & labels...
[2/4] Loading test data...
[3/4] Running predictions...
[4/4] Generating evaluation report...

                           precision  recall  f1-score  support
Apple___Apple_scab            0.97    0.96      0.97      634
Apple___Black_rot             0.99    0.98      0.98      553
...
Overall Accuracy : 96.42%
📄  Report saved → outputs/classification_report.txt
📊  Heatmap saved → outputs/confusion_matrix.png
```

---

### ✅ Step 5 – Run the Jupyter Notebook (Interactive)

```bash
# Install Jupyter if needed
pip install jupyter

# Launch
jupyter notebook notebooks/AgroDetect_Walkthrough.ipynb
```

Or open directly in VS Code with the **Jupyter extension**.

---

## 🔧 VS Code Extensions Recommended

| Extension | Purpose |
|-----------|---------|
| Python (Microsoft) | Linting, IntelliSense |
| Jupyter | Run .ipynb in VS Code |
| Pylance | Type checking |
| GitLens | Version control |

Install all at once:
```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
```

---

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Dataset | PlantVillage (54,306 images, 38 classes) |
| Base Model | MobileNetV2 (ImageNet weights) |
| Input Size | 224 × 224 RGB |
| Val Accuracy | ~96% |
| Inference Time | ~50ms / image (CPU) |

---

## 🗂️ Output Files After Full Run

```
outputs/
├── class_labels.json          ← {0: "Apple___Apple_scab", ...}
├── training_history.json      ← loss/accuracy per epoch
├── training_curves.png        ← accuracy & loss plots
├── prediction_result.png      ← bar chart of top-5 predictions
└── classification_report.txt  ← precision/recall/F1 per class

models/
└── agrodetect_mobilenet.h5    ← trained model (~9 MB)
```

---

*AgroDetect AI — Helping farmers detect crop disease early with deep learning.*
