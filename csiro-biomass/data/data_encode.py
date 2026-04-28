import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler

import os
from pathlib import Path

print("cwd =", os.getcwd())
print("__file__ =", Path(__file__).resolve().parent)

data_path = Path(__file__).resolve().parent

TARGET_COLUMNS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]

IMG_SIZE = (224, 224)




def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["image_path"] = df["image_path"].apply(
    lambda x: f"data/{x}"
    )

    print(df["image_path"][0])
    
    targets_df = (
        df.pivot(index="image_path", columns="target_name", values="target")
        .reset_index()
    )

    targets_df = targets_df[["image_path", *TARGET_COLUMNS]]
    return targets_df


def load_image_and_target(image_path, target):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, target


data = prepare_dataframe("data/train.csv")

image_paths = data["image_path"].to_numpy()
targets = data[TARGET_COLUMNS].to_numpy(dtype=np.float32)

# скейлим таргеты
scaler = StandardScaler()
targets = scaler.fit_transform(targets)

dataset = tf.data.Dataset.from_tensor_slices((data["image_path"].values, targets))
dataset = dataset.map(load_image_and_target, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)

for images, y in dataset.take(1):
    INPUT_SIZE = images.shape[1:]  # (224, 224, 3)
    print(images.shape)  # (batch, 224, 224, 3)
    print(y.shape)       # (batch, 5)
