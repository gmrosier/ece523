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

random.seed(1)


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)
cat_idx = class_names.index('cat')


# plotting helper function
def plot(img,label) :
    plt.imshow(img)
    plt.title(label)
    plt.show()

# extracts a rectangle from the essence of cat
def make_sticker (sticker_stats) :
    [ulx,uly,h,w] = sticker_stats
    x = np.copy(cat_img[uly:uly+h, ulx:ulx+w, :])
    return x

# applies sticker to the center of an image
def sticker_image (input_img, sticker) :
    sticker_shape = sticker.shape
    img_shape     = input_img.shape
    ulx = (img_shape[1] - sticker_shape[1]) // 2  # centering 
    uly = (img_shape[0] - sticker_shape[0]) // 2  # centering
    input_img[uly:uly+sticker_shape[0],ulx:ulx+sticker_shape[1],:] = sticker
    return input_img



############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################


# load saved model from memory
model = load_model('trained_models/cifar10_detector_model.h5')


# load GAN model to generate essensce of cat
gmodel = load_model('trained_models/cat_generator.hd5')
noise = np.random.normal(0, 1, (1, 100))


# generate cat image
cat_img = gmodel.predict(noise)[0]
plot(cat_img,'essence of cat to be used as a cat costume')


# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255 # convert to float
images = len(x_test)
x = np.copy(x_test[0:images])
y = y_test[0:images]

# pair down test images to only things the detector thought were non-cats, and therefore in need of a costume
y_pred = model.predict(x, verbose=1)
non_cat_idx = []
for i in range(images) :
    c = np.argmax(y_pred[i,:])
    if (not c==cat_idx) :
        non_cat_idx.append(i)

x_non_cat_test = x_test[non_cat_idx]
y_non_cat_test = y_test[non_cat_idx]
y = y_non_cat_test
images = len(x_non_cat_test)


# this has already been swept over the entire range
# for quicker repetition, the sweep is constrained to a region 
# alread demonstrated to be more effective
ulx_range = range(21,29)
uly_range = range(3,8)
wh_range  = range(3,11)

for wh in wh_range :
    best_cat_count = 0
    for ulx in ulx_range :
        for uly in uly_range :
            x = np.copy(x_non_cat_test)
            # create stickered images
            sticker = make_sticker([ulx,uly,wh,wh])
            for i in range(images) :
                x[i] = sticker_image(x[i],sticker)
            
            # run classifier with 
            y_pred = model.predict(x, verbose=1)

            cat_count = 0
            for i in range(images) :
                c = np.argmax(y_pred[i,:])
                if (c==cat_idx) : 
                    cat_count = cat_count + 1
                    #if ((not [i][0] == c) and (y_pred[i,c]>0.95)): plot(x[i],"image {:d}  {:s} wears the cat costume with probability {:0.3f})".format(i, class_names[y[i][0]], y_pred[i,c], ))
                #else : plot(x[i],"image {:d} still not a cat, instead {:s} with probability {:0.3f} (supposed to be {:s})".format(i, class_names[c], y_pred[i,c], class_names[y[i][0]]))
                    
            if (cat_count > best_cat_count) : 
                best_cat_count = cat_count
                best_sticker = [ulx,uly,wh,wh]

            print("{:0.3f} cat factor for sticker {:d}x{:d} from [{:d},{:d}]".format(cat_count/images,wh,wh,uly,ulx))

    print("\n\nbest cat factor was {:0.3f} for sticker".format(best_cat_count/images),best_sticker)
    plot(make_sticker(best_sticker),"best {:d}x{:d} cat costume ({:0.3f} effective)".format(wh,wh,best_cat_count/images))

    # debug printout to see how well it worked
    if (0) :
        # re-run best sticker to see where it worked the 
        x = np.copy(x_non_cat_test)
        # create stickered images
        sticker = make_sticker(best_sticker)
        for i in range(images) :
            x[i] = sticker_image(x[i],sticker)
        
        # run classifier with 
        y_pred = model.predict(x, verbose=1)

        for i in range(images) :
            c = np.argmax(y_pred[i,:])
            if (c==cat_idx and y_pred[i,c]>0.8 and (y[i][0]<2 or y[i][0]>7)) : 
                plot(x[i],"image {:d}  {:s} wears the cat costume with probability {:0.3f})".format(i, class_names[y[i][0]], y_pred[i,c] ))
            #else : plot(x[i],"image {:d} still not a cat, instead {:s} with probability {:0.3f} (supposed to be {:s})".format(i, class_names[c], y_pred[i,c], class_names[y[i][0]]))



