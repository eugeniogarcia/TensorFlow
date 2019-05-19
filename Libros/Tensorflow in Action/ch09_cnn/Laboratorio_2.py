# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:17:31 2019

@author: Eugenio
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

tf.enable_eager_execution()

print(tf.executing_eagerly()) 

W = tf.Variable(tf.random_normal([5, 5, 1, 32]))

print(W)

plt.figure()
rows, cols = 4, 8
for i in range(np.shape(W)[3]):
    img = W[:, :, 0, i]
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img, cmap='Greys_r', interpolation='none')
    plt.axis('off')

plt.show()