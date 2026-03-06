"""
AgroDetect AI - Plant Disease Classification Engine
====================================================
Step 3: Evaluate model performance (confusion matrix + report)
Command: python src/evaluate.py
"""

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONFIG ───────────────────────────────────
MODEL_PATH  = "models/agrodetect_mobilenet.h5"
DATA_DIR    = "data/processed"
LABELS_PATH = "outputs/class_labels.json"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

# ─── LOAD ─────────────────────────────────────
print("\n[1/4] Loading model & labels...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    class_labels = json.load(f)    # {"0": "Apple___Apple_scab", ...}
label_names = [class_labels[str(i)] for i in range(len(class_labels))]

# ─── DATA ─────────────────────────────────────
print("[2/4] Loading test data...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)

# ─── PREDICT ──────────────────────────────────
print("[3/4] Running predictions...")
preds       = model.predict(test_gen, verbose=1)
pred_labels = np.argmax(preds, axis=1)
true_labels = test_gen.classes

# ─── REPORT ───────────────────────────────────
print("\n[4/4] Generating evaluation report...\n")
report = classification_report(true_labels, pred_labels, target_names=label_names)
print(report)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix (top-15 classes for readability)
cm = confusion_matrix(true_labels, pred_labels)
top_n = min(15, len(label_names))
top_idx = np.argsort(cm.sum(axis=1))[-top_n:]
cm_top  = cm[np.ix_(top_idx, top_idx)]
names_top = [label_names[i].replace('___', '\n').replace('_', ' ')[:20] for i in top_idx]

fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(cm_top, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=names_top, yticklabels=names_top, ax=ax,
            linewidths=0.4, linecolor='#333')
ax.set_title('Confusion Matrix (Top Classes)', fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label',      fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

acc = np.sum(pred_labels == true_labels) / len(true_labels)
print(f"\n✅  Overall Accuracy : {acc*100:.2f}%")
print(f"📄  Report saved    → outputs/classification_report.txt")
print(f"📊  Heatmap saved   → outputs/confusion_matrix.png")
