"""
This file loads a pre-trained generator network from file
and generates cool kitty pictures. 


 _.-|   |          |\__/,|   (`\
{   |   |          |o o  |__ _) )
 "-.|___|        _.( T   )  `  /
  .--'-`-.     _((_ `^--' /_<  \
.+|______|__.-||__)`-'(((/  (((/

"""

import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.models import load_model
from keras.datasets import cifar10

# Load Model
model = load_model('trained_models/cat_generator.hd5')
model.summary()

# Build Random Noise Vector
noise = np.random.normal(0, 1, (1, 100))

# Generate One Cat Image and Display
cat_img = model.predict(noise)[0]
print(cat_img.shape)
plt.imshow(cat_img)
plt.show()
