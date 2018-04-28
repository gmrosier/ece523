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


# load saved model from memory
#model = load_model('trained_models/cifar10_detector_model.h5')
model = load_model('../saved_models/cifar10_ResNet20v1_model.176.h5')

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255 # convert to float
images = len(x_test)
x = np.copy(x_test[0:images])
y = y_test[0:images]

# pare down test images to only things the detector was confident were cats, and therefore in need of a disguise
y_pred = model.predict(x, verbose=1)
det_cat_idx = []
for i in range(images) :
    c = np.argmax(y_pred[i,:])
    if (c==cat_idx and y[i,0]==cat_idx and y_pred[i,c]>0.9) :
        #plot(x[i],"image {:d}:  {:s} ({:0.3f})".format(i,class_names[c],y_pred[i,c]))
        det_cat_idx.append(i)

x_cat_test = x_test[det_cat_idx]
y_cat_test = y_test[det_cat_idx]
y_pred_cat_test = y_pred[det_cat_idx]
initial_cat_score = y_pred_cat_test[:,cat_idx]
x = x_cat_test
y = y_cat_test
images = len(x_cat_test)

if (0) :
    for i in range(images) : 
        plot(x[i],"{:s} ({:0.3f})".format(class_names[y[i,0]],y_pred_cat_test[i,cat_idx]))


# make a random 3x3 pixel configuration
def make_sticker (color,val) :
    x = np.zeros([3,3,3],dtype='float32')
    
    for i in range(3) :
        for j in range(3) :
            if (val % 2) :
                x[i,j] = color
            val = val >> 1
    return x


# applies sticker to image, black pixels will be made transparent
def sticker_image (input_img, sticker, ulx, uly) :
    sticker_shape = sticker.shape
    for y in range(0,sticker_shape[0]) :
        for x in range(0,sticker_shape[1]) :
            if (not np.array_equal(sticker[y,x,:],[0,0,0])) :
                input_img[uly+y,ulx+x,:] = sticker[y,x]
    return input_img

# will try to apply sticker to the center area of the image
color = [0.0,0.0,0.5] # medium intensity perfect blue, should be good for cat disguise
ulx_range = range(12,19)
uly_range = range(12,19)
max_val   = 2**(3*3)
# this has already been characterized, so the val_range will be truncated to known good values for expediency
val_range = [1,59,438,463,495,503,510] #list(range(55,60)) + list(range(430,max_val)) # all binary combinations of the 3x3 pixel pattern
vals = len(val_range)
low_cat_score = np.ones([max_val,images])

total_reduction = np.zeros(max_val)
best_ulx        = np.zeros([max_val,images],dtype='int')
best_uly        = np.zeros([max_val,images],dtype='int')
best_reduction = 0
for val in val_range :
    sticker = make_sticker(color,val)
    #plot(sticker,"sticker {:d}".format(val))

    # sweep over a range of center pixels to find best location 
    for ulx in ulx_range :
        for uly in uly_range :
            # apply sticker to all cat images
            x = np.copy(x_cat_test) # avoid contaminating original data
            for i in range(images) :
                x[i] = sticker_image(x[i],sticker,ulx,uly)
            
            # run model
            y_pred = model.predict(x, verbose=0)

            # update low cat scores
            for i in range(images) :
                cat_score = y_pred[i,cat_idx]
                if (cat_score<low_cat_score[val,i]) :
                    low_cat_score[val,i] = cat_score
                    best_ulx[val,i] = ulx
                    best_uly[val,i] = uly

    best_disguised     = np.argpartition(low_cat_score[val], 1)[0]
    best_disguised_img = np.copy(x_cat_test[best_disguised])
    best_disguised_img = sticker_image(best_disguised_img,sticker,best_ulx[val,best_disguised],best_uly[val,best_disguised])
    best_disguised_pred = model.predict(np.reshape(best_disguised_img,[1,32,32,3]))
    c = np.argmax(best_disguised_pred[0])

    plot(best_disguised_img, "disguise {:d} reduced cat {:d} probability from {:0.3f} to {:0.3f} \nis now a {:s} with {:0.3f} probability".
       format(val,best_disguised,initial_cat_score[best_disguised],low_cat_score[val,best_disguised], class_names[c], best_disguised_pred[0,c] ))

    worst_disguised     =  np.argpartition(low_cat_score[val], -1)[0]
    worst_disguised_img = np.copy(x_cat_test[worst_disguised])
    worst_disguised_img = sticker_image(worst_disguised_img,sticker,best_ulx[val,worst_disguised],best_uly[val,worst_disguised])
    plot(worst_disguised_img, "disguise {:d} failed on cat {:d} probability went from {:0.3f} to {:0.3f}".format(val,worst_disguised,initial_cat_score[worst_disguised],low_cat_score[val,worst_disguised]))


    reductions = np.subtract(initial_cat_score,low_cat_score[val,:])
    total_reduction[val] = np.sum(reductions)

    print("disguise {:d} reduced overall average cat probability by {:0.3f}".format(val,total_reduction[val]/images))
    if (total_reduction[val] > best_reduction) : 
        best_reduction = total_reduction[val]
        best_val       = val

print("best disguise was {:d} with an average reduction of {:0.3f}".format(best_val,best_reduction/images))

#print("better disguises")
#topN = 10
#best_ind = np.argpartition(total_reduction, -topN)[-topN:]
#for i in best_ind :
#    print("{:d} = {:0.3f}".format(i,total_reduction[i]/images))

