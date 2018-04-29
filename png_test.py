#
# This script will determine the optimal cat costume sticker
# and quantify its effectiveness for different sizes
#

import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # disable GPU
import random
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
from matplotlib.pyplot import imread
random.seed(1)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)
cat_idx = class_names.index('cat')


# plotting helper function
def plot(img,label) :
    plt.imshow(img)
    plt.title(label)
    plt.show()

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################


# load saved model from memory
model = load_model('trained_models/cifar10_detector_model.h5')

# load kitteh and bunneh!
png = []
for image_path in ['test_images/bunny_with_cat_costume.png','test_images/kitten_with_disguise.png'] :
    png.append(imread(image_path))
x = np.asarray(png)


y_pred = model.predict(x, verbose=1)
for i in range(len(x)) :
    c = np.argmax(y_pred[i,:])
    plot(x[i],"{:s} ({:0.3f})".format(class_names[c],y_pred[i,c]))

