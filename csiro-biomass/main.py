import tensorflow as tf
from tensorflow import keras
from data.data_encode import IMG_SIZE
from model import build_model, INPUT_SIZE
import pandas as pd

TARGET_COLUMNS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]


model = build_model()
model.load_weights("best_model/best_model.h5")

image_bytes = tf.io.read_file("data/test/ID1001187975.jpg")
image = tf.image.decode_jpeg(image_bytes, channels=3)
image = tf.image.resize(image, IMG_SIZE)
image = tf.cast(image, tf.float32) / 255.0
image = tf.expand_dims(image, axis=0)

print(image.shape)

preds = model.predict(image, verbose=0)
pred = preds.flatten()

print(f"preds shape {pred.shape}")

df = pd.read_csv("data/sample_submission.csv")

df["target"] = pred

df.to_csv("submission.csv", index=False)