from model import model
from data.data_encode import dataset
import tensorflow as tf

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=["mae"],
)


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model/best_model.h5",
        monitor="loss",
        save_best_only=True,
        mode="min",
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=5,
        mode="min",
        restore_best_weights=True,
    ),

]

model.fit(
    dataset, 
    epochs=10, 
    callbacks=callbacks)