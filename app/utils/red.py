from app.config import pesos, umbrales
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
import cv2
import numpy as np
import asyncio
import json
ruta_archivo_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_red.txt')
ruta_imagenes = os.path.join(os.path.dirname(__file__), '..', 'config', 'imagenes')
ruta_save_keras = os.path.join(os.path.dirname(__file__), '..', 'config', 'model.h5')

def iniciar_red(total_salidas, total_entradas):
    pesos_red =  pesos.crear_pesos(total_entradas, total_salidas)
    umbrales_red = umbrales.crear_umbrales(total_salidas)
    return pesos_red, umbrales_red

def train_red(datos_entrada, datos_salida, patrones, total_salidas, total_entradas, iteraciones, rata_aprendizaje, error_maximo, pesos_red, umbrales_red):
    error_iteration = []
    red_is_train = False
    for iteration in range(iteraciones):
        errores_permitidos = 0
        #iteramos cada patron
        for patron in range(patrones):
            salidas_neuronas = []
            errores = []
            error_permitido = 0
            #iteracion de las salidas
            for i in range(total_salidas):
                suma_entrada_pesos = 0
                for j in range(total_entradas):
                    suma_entrada_pesos += ((datos_entrada[patron][j] * pesos_red[j][i]))
                salida_neurona = suma_entrada_pesos - umbrales_red[i]
                if salida_neurona > 0:
                    salida_neurona = 1
                else:
                    salida_neurona = 0
                salidas_neuronas.append(salida_neurona)
                #calculamos el error 
                error = datos_salida[patron][i] - salidas_neuronas[i]
                errores.append(error)
                error_permitido += error
            # ajustamos umbrales y pesos
            for i in range(total_salidas):
                for j in range(total_entradas):
                    pesos_red[j][i] = pesos_red[j][i] + (rata_aprendizaje * errores[i] * datos_entrada[patron][j])
                umbrales_red[i] = umbrales_red[i] + (rata_aprendizaje * errores[i] * datos_salida[patron][i])
            #sumamos los errores permitidos por cada patron de la iteracion
            errores_permitidos += (abs(error_permitido) / total_salidas)
        #validamos el si el error permitido es menor al error maximo para ver si concluimos el entrenamiento.
        error_iteracion = (errores_permitidos / patrones)
        error_iteration.append(error_iteracion)
        if (error_iteracion <= error_maximo):
            #guardamos los umbrales y pesos en una matriz  
            pesos.guardar_pesos(pesos_red)
            umbrales.guardar_umbrales(umbrales_red)
            red_is_train = True
            with open(ruta_archivo_config, 'w') as f:
                f.write('1;1')
            break
    return red_is_train, error_iteration



def simular_keras(contents, connected_clients):
    class_labels = ['gafas', 'mando', 'mouse']
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
    img = cv2.cvtColor(np.array(contents), cv2.COLOR_BGR2GRAY)
    print(img.shape)

    # Preprocesar la imagen (ajustar tamaño, normalizar, etc.)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)


    # Realizar la predicción utilizando el modelo
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    #asyncio.get_event_loop().run_until_complete(send_predictions(predictions))

    # Devolver las predicciones como respuesta
    return json.dumps({"predictions": predicted_class})

async def send_predictions(predictions, connected_clients):
    for client in connected_clients:
        await client.send_text(predictions)

def entrenar_keras():
    model, train_data, test_data = iniciar_keras()

    max_error = 0.01  # Error máximo deseado

    loss_values = []
    validation_loss = []

    best_weights = None
    best_validation_loss = float('inf')

    for epoch in range(500):
        # Realizar una época de entrenamiento
        history = model.fit(train_data, epochs=1, validation_data=test_data)

        # Registrar las pérdidas
        loss_values.append(history.history['loss'][0])
        validation_loss.append(history.history['val_loss'][0])

        # Verificar si se alcanza el error máximo
        if validation_loss[-1] <= max_error:
            print(validation_loss, max_error)
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

    return loss_values, validation_loss



def iniciar_keras():
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Configurar el generador de flujo de datos de entrenamiento y prueba   
    train_generator = train_datagen.flow_from_directory(
        ruta_imagenes,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale'
    )

    test_generator = test_datagen.flow_from_directory(
        ruta_imagenes,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale'
    )

    num_classes = train_generator.num_classes
    def custom_loss(y_true, y_pred):
        # Establecer el umbral máximo de error
        umbral_error_maximo = 0.01
        # Calcular la pérdida utilizando categorical_crossentropy
        loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        # Calcular el error medio
        error = tf.reduce_mean(loss)

        # Aplicar una penalización si el error excede el umbral
        penalized_loss = tf.cond(error > umbral_error_maximo, lambda: loss + 1.0, lambda: loss)

        return penalized_loss
    # Construir el modelo
    model = keras.Sequential([
        keras.layers.MaxPooling2D((2, 2), input_shape=(64, 64, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # tasa de aprendizaje
    learning_rate = 0.001  # Tasa de aprendizaje deseada
    optimizer = Adam(learning_rate=learning_rate)
    # Compilar el modelo
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
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

