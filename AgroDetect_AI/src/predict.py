"""
AgroDetect AI - Plant Disease Classification Engine
====================================================
Step 2: Run inference on a single leaf image
Command: python src/predict.py --image path/to/leaf.jpg
"""

import argparse
import json
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="AgroDetect AI – Predict plant disease from a leaf image")
parser.add_argument("--image",  required=True,               help="Path to leaf image")
parser.add_argument("--model",  default="models/agrodetect_mobilenet.h5", help="Path to trained model")
parser.add_argument("--labels", default="outputs/class_labels.json",      help="Path to class labels JSON")
parser.add_argument("--top",    type=int, default=5,          help="Number of top predictions to show")
args = parser.parse_args()

# ─────────────────────────────────────────────
# LOAD MODEL & LABELS
# ─────────────────────────────────────────────
print(f"\n[1/4] Loading model from: {args.model}")
model = tf.keras.models.load_model(args.model)

print(f"[2/4] Loading class labels from: {args.labels}")
with open(args.labels) as f:
    class_labels = json.load(f)   # {0: 'Apple___Apple_scab', ...}

# ─────────────────────────────────────────────
# PREPROCESS IMAGE
# ─────────────────────────────────────────────
print(f"[3/4] Preprocessing image: {args.image}")

img_bgr  = cv2.imread(args.image)
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_res  = cv2.resize(img_rgb, (224, 224))
img_norm = img_res / 255.0
img_inp  = np.expand_dims(img_norm, axis=0)   # (1, 224, 224, 3)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
print(f"[4/4] Running inference...")

preds      = model.predict(img_inp, verbose=0)[0]
top_idx    = np.argsort(preds)[::-1][:args.top]
top_labels = [class_labels[str(i)] for i in top_idx]
top_probs  = [float(preds[i]) for i in top_idx]

# ─────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  🌿 AgroDetect AI – Prediction Results")
print("="*55)
for rank, (label, prob) in enumerate(zip(top_labels, top_probs), 1):
    bar = "█" * int(prob * 30)
    print(f"  #{rank}  {label:<35}  {prob*100:5.1f}%  {bar}")
print("="*55)
print(f"\n✅ Top Prediction: {top_labels[0]}  ({top_probs[0]*100:.1f}% confidence)\n")

# ─────────────────────────────────────────────
# SAVE RESULT PLOT
# ─────────────────────────────────────────────
fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')

ax_img.imshow(img_rgb)
ax_img.set_title("Input Leaf Image", color='white', fontsize=13, fontweight='bold')
ax_img.axis('off')

colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_labels))]
bars   = ax_bar.barh(range(len(top_labels)), top_probs, color=colors, edgecolor='white', linewidth=0.5)
ax_bar.set_yticks(range(len(top_labels)))
ax_bar.set_yticklabels([l.replace('_', ' ') for l in top_labels], color='white', fontsize=9)
ax_bar.set_xlabel('Confidence', color='white')
ax_bar.set_title('Top Predictions', color='white', fontsize=13, fontweight='bold')
ax_bar.tick_params(colors='white')
ax_bar.set_facecolor('#16213e')
ax_bar.spines['bottom'].set_color('#444')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_color('#444')
ax_bar.invert_yaxis()

for bar_obj, prob in zip(bars, top_probs):
    ax_bar.text(bar_obj.get_width() + 0.005, bar_obj.get_y() + bar_obj.get_height()/2,
                f'{prob*100:.1f}%', va='center', color='white', fontsize=9)

plt.tight_layout()
out_path = "outputs/prediction_result.png"
plt.savefig(out_path, dpi=150, facecolor='#1a1a2e')
plt.close()
print(f"📊 Result chart saved → {out_path}")
