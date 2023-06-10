import glob
from app.config import pesos, umbrales
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import cv2
import numpy as np
import json
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.metrics import mean_squared_error
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from PIL import Image

ruta_archivo_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_red.txt')
ruta_imagenes = os.path.join(os.path.dirname(__file__), '..', 'config', 'imagenes')
ruta_save_keras = os.path.join(os.path.dirname(__file__), '..', 'config', 'model.h5')
ruta_save_perceptron = os.path.join(os.path.dirname(__file__), '..', 'config', 'perceptron_model.pkl')


def cantidad_imagenes():
    subdirectorios = [nombre for nombre in os.listdir(ruta_imagenes) if os.path.isdir(os.path.join(ruta_imagenes, nombre))]
    total_patrones = 0
    for subdirectorio in subdirectorios:
        ruta_subdirectorio = os.path.join(ruta_imagenes, subdirectorio)
        elementos = len(os.listdir(ruta_subdirectorio))
        total_patrones += elementos
        print(f"Cantidad de elementos en {subdirectorio}: {elementos}")
    return {"patrones": total_patrones, "categorias": len(subdirectorios)}

def iniciar_red(total_salidas, total_entradas):
    pesos_red =  pesos.crear_pesos(total_entradas, total_salidas)
    umbrales_red = umbrales.crear_umbrales(total_salidas)
    return pesos_red, umbrales_red

def extraer_caracteristicas(image_paths):
    features = []
    for image_path in image_paths:
        # Cargar la imagen utilizando OpenCV
        image = cv2.imread(image_path)
        # Verificar si la imagen se cargó correctamente
        if image is None:
            # Si la imagen no se cargó, pasar a la siguiente
            continue
        # Puedes ajustar el tamaño de las imágenes según tus necesidades
        image = cv2.resize(image, (128, 128))
        # Convertir la imagen a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calcular el descriptor HOG
        hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        # Agregar las características a la lista
        features.append(hog_features)
    # Convertir la lista de características en un array de NumPy
    return features


def simular_keras(contents, connected_clients):
    class_labels = ['Gafas', 'Lapiz', 'Maceta', 'Mouse', 'Portatil', 'Teclado']
    def custom_loss(y_true, y_pred):
        # Establecer el umbral máximo de error
        umbral_error_maximo = 0.2

        # Calcular la pérdida utilizando categorical_crossentropy
        loss = keras.losses.categorical_crossentropy(y_true, y_pred)

        # Calcular el error medio
        error = tf.reduce_mean(loss)

        # Aplicar una penalización si el error excede el umbral
        penalized_loss = tf.cond(error > umbral_error_maximo, lambda: loss + 1.0, lambda: loss)

        return penalized_loss


    # Definir el ámbito de objetos personalizados de Keras
    with tf.keras.utils.custom_object_scope({'custom_loss': custom_loss}):
        # Cargar el modelo
        model = load_model(ruta_save_keras)
    # Convertir los datos de la imagen a matriz numpy
    #nparr = np.frombuffer(contents, np.uint8)
    img = cv2.cvtColor(np.array(contents), cv2.COLOR_RGB2RGBA)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Preprocesar la imagen (ajustar tamaño, normalizar, etc.)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)


    # Realizar la predicción utilizando el modelo
    predictions = model.predict(img)
    
    predicted_class_index = np.argmax(predictions[0])
    print(predicted_class_index)
    predicted_class = class_labels[predicted_class_index]
    #asyncio.get_event_loop().run_until_complete(send_predictions(predictions))

    # Devolver las predicciones como respuesta
    return json.dumps({"predictions": predicted_class})

async def send_predictions(predictions, connected_clients):
    for client in connected_clients:
        await client.send_text(predictions)

# Función de carga personalizada para leer las imágenes
def custom_loader(image_path):
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        return image.convert('RGB')
async def entrenar_perceptron(iteraciones, error_maximo, tasa_aprendizaje, websocket):
    return ""


async def entrenar_keras(iteraciones, error_maximo, tasa_aprendizaje, websocket):
    model, train_data, test_data = iniciar_keras(tasa_aprendizaje)

    max_error = error_maximo  # Error máximo deseado

    loss_values = []
    validation_loss = []

    best_weights = None
    best_validation_loss = float('inf')
    total_epoch = 0
    error_response = []
    print("Entrenando")
    for epoch in range(iteraciones):
        # Realizar una época de entrenamiento
        total_epoch += 1
        history = model.fit(train_data, epochs=1, validation_data=test_data)

        # Registrar las pérdidas
        loss_values.append(history.history['loss'][0])
        json_loss = {
            "name": "Iteracion " + str(epoch + 1),
            "Error": history.history['val_loss'][0],
            
        }
        error_response.append(json_loss)
        validation_loss.append(history.history['val_loss'][0])

        json_return = json.dumps({"lost_response": error_response, "finish": 0, "iteration": (epoch + 1), "iterations": iteraciones})
        await websocket.send_text(json_return)
        # Verificar si se alcanza el error máximo
        if validation_loss[-1] <= max_error:
            break

        # Actualizar los mejores pesos y la mejor pérdida de validación
        if validation_loss[-1] < best_validation_loss:
            best_weights = model.get_weights()
            best_validation_loss = validation_loss[-1]

    # Verificar si se encontraron los mejores pesos
    if best_weights is not None:
        # Restaurar los mejores pesos obtenidos durante el entrenamiento
        model.set_weights(best_weights)

    model.save(ruta_save_keras)

    return error_response, total_epoch


def dataframe_images(red):
    # Listas para almacenar las rutas de las imágenes y las etiquetas
    image_paths = []
    labels = []

    # Extensiones de archivo permitidas
    extensiones_permitidas = ['.jpg', '.jpeg', '.png']
    for label in os.listdir(ruta_imagenes):
        label_dir = os.path.join(ruta_imagenes, label)
        if os.path.isdir(label_dir):
            # Recorrer los archivos de imagen en el subdirectorio
            for file in os.listdir(label_dir):
                extension = os.path.splitext(file)[1].lower()
                if extension in extensiones_permitidas:
                    # Construir la ruta completa de la imagen
                    image_path = os.path.join(label_dir, file)
                    
                    # Agregar la ruta de la imagen a la lista
                    image_paths.append(image_path)
                    
                    # Agregar la etiqueta a la lista
                    labels.append(label)

    # Crear un dataframe con las rutas de las imágenes y las etiquetas
    data = pd.DataFrame({'image_paths': image_paths, 'labels': labels})
    labels = ''
    if red == 'keras':
        labels = data['labels']
    else:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(data['labels'])
    # Dividir los conjuntos de entrenamiento y prueba
    train_data, test_data, train_labels, test_labels = train_test_split(data['image_paths'], labels, test_size=0.2, random_state=42)
    return train_data, test_data, train_labels, test_labels, data


def datagen_dataframe(datagen, data, labels):
    return datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'image_paths': data, 'labels': labels}),
        x_col='image_paths',
        y_col='labels',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
def iniciar_keras(tasa_aprendizaje):
    train_data, test_data, train_labels, test_labels, data = dataframe_images('keras')

    # Preprocesamiento de imágenes
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Generadores de datos de entrenamiento y prueba
    train_generator = datagen_dataframe(train_datagen, train_data, train_labels)
    test_generator = datagen_dataframe(test_datagen, test_data, test_labels)

    # Cargar el modelo preentrenado
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Congelar las capas del modelo base para que no se entrenen durante la transferencia de aprendizaje
    for layer in base_model.layers:
        layer.trainable = False

    # Construir el modelo de la red personalizada
    model = keras.models.Sequential()
    model.add(base_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(data['labels'].unique()), activation='softmax'))


    # tasa de aprendizaje
    learning_rate = tasa_aprendizaje  # Tasa de aprendizaje deseada
    optimizer = Adam(learning_rate=learning_rate)
    # Compilación del modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, train_generator, test_generator

def simular(entradas, pesos_red, umbrales_red, total_patrones, total_salidas, total_entradas):
    salidas = []
    for patron in range(total_patrones):
        salidas_neuronas = []
        for i in range(total_salidas):
            suma_entrada_pesos = 0
            # iteramos cada elemento del patron
            for j in range(total_entradas):
                suma_entrada_pesos += ((entradas[patron][j] * pesos_red[j][i]))
            salida_neurona = suma_entrada_pesos - umbrales_red[i]
            if salida_neurona > 0:
                salida_neurona = 1
            else:
                salida_neurona = 0
            salidas_neuronas.append(salida_neurona)
        salida =  " ".join(str(x) for x in salidas_neuronas)
        salidas.append(salida)
    return salidas


# Definir la arquitectura de la red
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 28 * 28, 10)  # Ajustar el tamaño de entrada según tus datos

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

