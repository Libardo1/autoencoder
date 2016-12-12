#coding=utf8

from keras.layers import Input, Dense
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

encoding_dim = 32
image_width = 28
input_dim = image_width ** 2
epoch = 50

autoencoder = Sequential()
autoencoder.add(Dense(encoding_dim, input_dim=input_dim, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
# x_train is a 60000 * 28 * 28 array of uint (0-255)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# 归一化
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
# np.prod: 矩阵内所有元素乘积, x_train.shape = [60000, 28, 28]

print x_train.shape # [60000, 784]
print x_test.shape  # [10000, 784]

autoencoder.fit(x_train, x_train, nb_epoch = epoch, validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)
n = 10
for i in range(n):
    plt.subplot(2, n, i + 1)
    # subplot(nrows, ncols, plot_number)
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot
    plt.imshow(x_test[i].reshape(image_width, image_width))

    plt.subplot(2, n, n + i + 1)
    plt.imshow(decoded_imgs[i].reshape(image_width, image_width))
plt.show()


