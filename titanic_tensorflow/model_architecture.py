import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data import train_data, train_targets
from tensorflow.keras import regularizers


input = keras.Input(shape=(train_data.shape[1],)) # количество признаков в данных
# x = layers.Dense(256, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(0.01))(input) # выключаем смещение, т.к. его в себя включает batchnormalization()
# x = layers.BatchNormalization()(x) # добавляем слой нормализации, чтобы ускорить обучение и улучшить стабильность модели
# x = layers.Dense(128, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(0.01))(x)
# x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation='relu', use_bias = False, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(input)
x = layers.BatchNormalization()(x)
x = layers.Dense(32, activation='relu', use_bias = False, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(16, activation='relu', use_bias = False, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
x = layers.BatchNormalization()(x)
droput = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(droput) # так как у нас бинарная классификация, используем сигмоиду
model = keras.Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='titanic_tensorflow\\best_model\\best_model_titanic.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )
]