
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Project settings
DATASET_PATH = "dataset"
MODEL_PATH = "models/waste_classifier.keras"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 8

print("Loading waste image dataset...")

# Load training and validation images
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names
print("Waste categories:", class_names)

# Improve data loading speed
train_data = train_data.prefetch(tf.data.AUTOTUNE)
val_data = val_data.prefetch(tf.data.AUTOTUNE)

# Load MobileNetV2 pretrained model
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Build classification model
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting model training...")

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Evaluate model
loss, accuracy = model.evaluate(val_data)

print("\nTraining completed!")
print(f"Validation accuracy: {accuracy * 100:.2f}%")

# Save trained model
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

# Save category names
with open("models/class_names.json", "w") as file:
    json.dump(class_names, file)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
