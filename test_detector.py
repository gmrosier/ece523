"""
Loads the specified pre-trained model and tests it against the
CIFAR-10 test data.

This file was created to test how to load a file from a file.
"""

import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10

# Load Model
model = load_model('trained_models/cifar10_detector_model.h5')
model.summary()

# Load Data
_, (x_test, y_test) = cifar10.load_data()

# Convert from uint8 and normalize
x_test = x_test.astype('float32') / 255

# Convert to Multi-Category Data
y = keras.utils.to_categorical(y_test, 10)

# Score Training
scores = model.evaluate(x_test, y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Test One Cat Image
cat_idx =  y_test.flatten().tolist().index(3)
cat_img = x_test[cat_idx,:,:,:]
pred = model.predict(np.reshape(cat_img, (1,32,32,3)))[0]
print(pred)
print("Pred Cat: {}".format(pred[3]*100))
plt.imshow(cat_img)
plt.show()
