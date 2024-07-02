import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np

model_path = os.path.join('model', 'mnist_model.h5')
data_path = os.path.join('data', 'train_data.npz')

# Inicializar la aplicación Flask
app = Flask(__name__, static_folder='../static', template_folder='../templates')

# Cargar el modelo de Keras y recrear el optimizador
model = load_model(model_path)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Cargar datos acumulados
if os.path.exists(data_path):
    data = np.load(data_path)
    train_images = data['images']
    train_labels = data['labels']
else:
    train_images = np.empty((0, 28, 28, 1))
    train_labels = np.empty((0, 10))

# Definir la ruta para cargar el formulario HTML
@app.route('/')
def index():
    return render_template('index.html')

# Definir la ruta para predecir el dígito
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))
    image = np.array(image) / 255.0  # Normalizar
    image = image.reshape(1, 28, 28, 1)
    
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    
    return jsonify({'digit': int(digit)})

# Definir la ruta para corregir la predicción y entrenar el modelo
@app.route('/correct', methods=['POST'])
def correct():
    global train_images, train_labels

    file = request.files['file']
    correct_digit = int(request.form['correctDigit'])
    
    # Preprocesar la imagen de manera similar
    image = Image.open(file).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(28, 28, 1)  # Eliminar la dimensión extra

    # Convertir la etiqueta correcta a one-hot encoding
    label = to_categorical(correct_digit, num_classes=10)
    
    # Acumular la nueva imagen y etiqueta
    train_images = np.append(train_images, [image], axis=0)
    train_labels = np.append(train_labels, [label], axis=0)
    
    # Entrenar el modelo con el nuevo dato
    model.fit(np.array([image]), np.array([label]), epochs=1, verbose=0)
    
    # Guardar el modelo actualizado y los datos acumulados
    model.save(model_path)
    np.savez(data_path, images=train_images, labels=train_labels)
    
    return jsonify({'status': 'success'})

# Definir la ruta para reentrenar el modelo con todos los datos acumulados
@app.route('/retrain', methods=['POST'])
def retrain():
    global train_images, train_labels

    if train_images.shape[0] > 0:
        model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=1)
        model.save(model_path)
        return jsonify({'status': 'retrained'})
    else:
        return jsonify({'status': 'no data'})

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)
