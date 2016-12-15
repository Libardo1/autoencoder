#coding=utf8

from keras.layers import *
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

image_width = 28
epoch = 50

autoencoder = Sequential()
autoencoder.add(Convolution2D(32, 3, 3, input_shape=(1, image_width, image_width),
                              border_mode='same', activation='relu'))
# Convolution2D(卷积核数目，卷积核宽，卷积核高，...)
# border_mode='same' 或'valid'，前者输入和输出一样大，后者会“缩水一圈”
autoencoder.add(MaxPooling2D((2, 2), border_mode='same'))
# 下采样，长宽分别缩小一半
autoencoder.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
autoencoder.add(MaxPooling2D((2, 2), border_mode='same'))
autoencoder.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid'))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
# x_train is a 60000 * 28 * 28 array of uint (0-255)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# 归一化
x_train = x_train.reshape(len(x_train), 1, image_width, image_width)
x_test = x_test.reshape(len(x_test), 1, image_width, image_width)
# x_train.shape = [60000, 1, 28, 28]

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

print x_train.shape # [60000, 1, 28, 28]
print x_test.shape  # [10000, 1, 28, 28]

autoencoder.fit(x_train_noisy, x_train, nb_epoch = epoch, validation_data=(x_test_noisy, x_test))

decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
for i in range(n):
    plt.subplot(2, n, i + 1)
    # subplot(nrows, ncols, plot_number)
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot
    plt.imshow(x_test_noisy[i].reshape(image_width, image_width))

    plt.subplot(2, n, n + i + 1)
    plt.imshow(decoded_imgs[i].reshape(image_width, image_width))
plt.show()


