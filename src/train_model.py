import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

# Definir la ruta base a la carpeta de datos
data_path = 'data'

# Leer los archivos IDX
def read_idx(filename):
    import struct
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Ajustar las rutas para incluir la carpeta 'data'
train_images = read_idx(os.path.join(data_path, 'train-images.idx3'))
train_labels = read_idx(os.path.join(data_path, 'train-labels.idx1'))
test_images = read_idx(os.path.join(data_path, 't10k-images.idx3'))
test_labels = read_idx(os.path.join(data_path, 't10k-labels.idx1'))

# Normalizar las imágenes y convertir las etiquetas a categorías
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Definir el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Guardar el modelo
model.save('mnist_model.h5')
