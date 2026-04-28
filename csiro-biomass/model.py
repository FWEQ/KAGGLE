import tensorflow as tf
from tensorflow import keras
from keras import layers

from data.data_encode import dataset, INPUT_SIZE

def build_model():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            ]
    )

    base = keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=INPUT_SIZE,
        weights="imagenet",
    )

    input_layer = keras.Input(shape=INPUT_SIZE)
    x = data_augmentation(input_layer)

    x = base(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)  

    output = layers.Dense(5, activation="linear")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model

model = build_model()    
model.summary()