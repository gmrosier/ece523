# Train Detector on CIFAR-10 Dataset to detect cats
# In the cifar dataset cats are label 4

import os
import sys

from matplotlib import pyplot as plt
import tensorflow as tf
from official.resnet import resnet_model
from official.resnet import resnet_run_loop

import cifar10 as cf

data = cf.CifarDataset('.')

for img in data:
  if (img['catagory'] == 'cat'):
    print(img['catagory'])
    plt.imshow(img['image'], interpolation='bicubic')
    plt.show()