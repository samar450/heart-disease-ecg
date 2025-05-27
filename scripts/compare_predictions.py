import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Load model
model = tf.keras.models.load_model("models/heart_disease_model_binary.keras")

# Settings
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    "train_binary",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_data = datagen.flow_from_directory(
    "test_binary",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

def evaluate(name, data):
    preds = model.predict(data)
    y_true = data.classes
    y_pred = (preds > 0.5).astype(int)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))

evaluate("Training", train_data)
evaluate("Test", test_data)