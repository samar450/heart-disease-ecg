import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
test_dir = "test_binary"
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model("models/heart_disease_model_binary.keras")

# Test data generator
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")