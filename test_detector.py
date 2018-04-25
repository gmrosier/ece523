import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.models import load_model
from keras.datasets import cifar10

# Load Model
model = load_model('trained_models/cat_generator.h5')
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
print("Pred Cat: {}".format(model.predict(np.reshape(cat_img, (1,32,32,3)))[3]*100))
plt.imshow(cat_img)
plt.show()
