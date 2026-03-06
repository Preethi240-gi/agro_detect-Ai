"""
AgroDetect AI - Plant Disease Classification Engine
====================================================
Step 1: Train the MobileNet transfer learning model
Command: python src/train.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
NUM_CLASSES = 38        # PlantVillage has 38 classes
DATA_DIR    = "data/processed"
MODEL_OUT   = "models/agrodetect_mobilenet.h5"
HISTORY_OUT = "outputs/training_history.json"
PLOT_OUT    = "outputs/training_curves.png"

# ─────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────
print("\n[1/5] Loading data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class labels
class_labels = {v: k for k, v in train_gen.class_indices.items()}
with open("outputs/class_labels.json", "w") as f:
    json.dump(class_labels, f, indent=2)
print(f"    Classes found: {len(class_labels)}")

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
print("\n[2/5] Building MobileNetV2 transfer learning model...")

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"    Total params: {model.count_params():,}")
print(f"    Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
print("\n[3/5] Setting up callbacks...")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

callbacks = [
    ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n[4/5] Training model...")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# Save history
with open(HISTORY_OUT, "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
print("\n[5/5] Saving training curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train Acc',  color='#2ecc71')
ax1.plot(history.history['val_accuracy'], label='Val Acc',    color='#27ae60', linestyle='--')
ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend()

ax2.plot(history.history['loss'],     label='Train Loss', color='#e74c3c')
ax2.plot(history.history['val_loss'], label='Val Loss',   color='#c0392b', linestyle='--')
ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.legend()

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150)
plt.close()

print(f"\n✅ Training complete!")
print(f"   Model saved  → {MODEL_OUT}")
print(f"   Curves saved → {PLOT_OUT}")
