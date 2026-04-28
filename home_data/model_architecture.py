import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data import train_data, train_targets
from tensorflow.keras import regularizers

# архитектура для задачи регрессии, так как у нас числовая целевая переменная, которая может принимать любые значения
input = keras.Input(shape=(train_data.shape[1],)) # количество признаков в данных
# x = layers.Dense(512, activation='relu', use_bias=False, kernel_initializer='he_normal')(input)
# x = layers.BatchNormalization()(x)
# x = layers.Dense(256, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(input)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)

x = layers.Dense(128, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(input)
x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)

x = layers.Dense(64, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.2)(x)

x = layers.Dense(32, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

x = layers.Dense(16, activation='relu', use_bias=False, kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
x = layers.BatchNormalization()(x)
#dropout = layers.Dropout(0.5)(x)
output = layers.Dense(1)(x) # так как у нас регрессия, то выходной слой без активации
model = keras.Model(inputs=input, outputs=output)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='home_data\\best_model\\best_model_homes.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    ),

    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # уменьшаем в 2 раза
    patience=5,       # если нет улучшения 5 эпох
    min_lr=1e-6
    ),
    keras.callbacks.TensorBoard(
    log_dir = 'C:\\vs_code_projects\\KERAS+TENSOR\\venv\\guidance\\logs'
    ),
]

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])