import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Definir y compilar el modelo
def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cargar los datos de MNIST
(train_images, train_labels), (val_images, val_labels) = mnist.load_data()

# Preprocesar los datos
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
val_images = val_images.reshape(-1, 28, 28, 1) / 255.0
train_labels = to_categorical(train_labels, num_classes=10)
val_labels = to_categorical(val_labels, num_classes=10)

# Crear y entrenar el modelo
model = create_model()
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, batch_size=32)

# Guardar el modelo entrenado
model_path = os.path.join('model', 'mnist_model.h5')
model.save(model_path)

# Guardar el historial de entrenamiento
history_path = os.path.join('data', 'train_history.npz')
np.savez(history_path, 
         accuracy=history.history['accuracy'], 
         val_accuracy=history.history['val_accuracy'], 
         loss=history.history['loss'], 
         val_loss=history.history['val_loss'])

print("Entrenamiento completo y historial guardado.")
