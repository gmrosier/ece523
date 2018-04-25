import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.models import load_model
from keras.datasets import cifar10

# Load Model
model = load_model('trained_models/cat_generator.hd5')
model.summary()

# Build Noise Vector
noise = np.random.normal(0, 1, (1, 100))

# Generate One Cat Image
cat_img = model.predict(noise)[0]
print(cat_img.shape)
plt.imshow(cat_img)
plt.show()
