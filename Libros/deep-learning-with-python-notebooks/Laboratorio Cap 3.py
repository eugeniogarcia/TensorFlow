# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:14:30 2019

@author: Eugenio
"""

import numpy as np
import matplotlib.pyplot as plt

#Usa un dataset de los embebidos en Keras
from keras.datasets import imdb
#Define el modelo
from keras import models
from keras import layers
#Define el optimizador, funciones de error
from keras import optimizers
#Para poder definir funciones de error y metricas custom
from keras import losses
from keras import metrics


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("*"*20)
print("25000 comentarios")
print(train_data.shape)
print(len(train_data))
print("Cada comentario es una lista con un numero variable de palabras - codigicadas como int32")
print(len(train_data[0]))
print("*"*20)
word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print("*"*20)
print("Ejemplo de una revision")
print(decoded_review)
print("*"*20)


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')



def vectorize_sequences(sequences, dimension=10000):
    #Crea una matriz numeroDeComentarios x 10000
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #Para cada comentario la columna tendra 1 en aquellas palabras presentes en el comentario
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print("*"*20)
print("Definimos el modelo")

model = models.Sequential()

#La entrada es un vector de 10000 posiciones
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print("*"*20)
print("Definimos el optimizador, la funcion de error, y la metrica que vamos a seguir")

#Optimizador por defecto
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#Personaliza el optimizador
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

#Define una funcion de error y una metrica custom
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

print("*"*20)
print("Vamos a separar el training set en dos conjuntos; Uno lo usaremos para entrenar, y el otro para verificar")
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#20 epocs, con batches de 512. Cada epoch tendra tantas iteracciones 
#como se preciso para cubrir todos los datos en bloques de batch-size
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))

print("*"*20)
print("Vemos que metricas tenemos disponibles")
history_dict = history.history
print(history_dict.keys)

print("*"*20)
print("Hacemos una representacion grafica de las metricas")

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, 20)
plt.subplot(2,1,1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.subplot(2,1,2)
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
