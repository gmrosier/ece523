import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.models import load_model, Model
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.datasets import cifar10

# Load Model
base_model = load_model('trained_models/cifar10_detector_model.h5')

# Only Return Cat
y = Lambda(lambda x: x[:, 3], output_shape=lambda x: (x[0],))(base_model.output)
model = Model(base_model.input, y)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.00005),
              metrics=['accuracy'])

# Load Data
_, (x_test, y_test) = cifar10.load_data()

# Convert from uint8 and normalize
x_test = x_test.astype('float32') / 255

## Convert to Single Category Data
y_test = y_test.flatten()
y_test[y_test != 3] = 0
y_test[y_test == 3] = 1
print(y_test.shape)

# Score Training
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Test One Cat Image
cat_idx =  y_test.flatten().tolist().index(1)
cat_img = x_test[cat_idx,:,:,:]
print("Pred Cat: {}".format(model.predict(np.reshape(x_test[:3], (3,32,32,3)))*100))
plt.imshow(cat_img)
plt.show()
