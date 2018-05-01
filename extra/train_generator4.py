# This is build from the following paper
#
# One pixel attack for fooling deep neural networks
# Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai
# http://arxiv.org/abs/1710.08864

import time
import pickle
from functools import partial
import numpy as np
from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Lambda
from keras.datasets import cifar10

def modify(x, image, box_size):
  pixels = len(x)//2
  shape = (pixels, 2)
  x = np.reshape(x, shape)
  
  # Copy Image
  image = np.copy(image)

  # Modify The Pixes
  for i in range(pixels):
    x_idx, y_idx = x[i, :]
    x_idx = int(np.floor(x_idx))
    y_idx = int(np.floor(y_idx))

    for bx in range(box_size):
      for by in range(box_size):
        image[x_idx+bx, y_idx+by] = [0.0, 0.0, 1.0]
      
  return image


def predict(x, image, detector, box_size):
  # Modifiy the specified pixels
  mimg = modify(x, image, box_size)
  
  # Test New Image
  pred = detector.predict(np.reshape(mimg, (1,32,32,3)))[0][3]

  # Return Prediction
  return pred


def success(x, convergence=1.0, image=None, detector=None, box_size=None):

  # Modifiy the specified pixels
  mimg = modify(x, image, box_size)
  pred = detector.predict(np.reshape(mimg, (1,32,32,3)))[0]
  
  # Return if image is no longer a cat
  retVal = False
  c = pred[3]
  for p in pred:
    if (p > c):
      retVal = True

  if retVal:
    print("Found Solution")

  return retVal


def find_change(image, detector, pixel_count=1, max_iter=10, pop_size=200, box_size=1):

  # Get Image Shape
  shape = image.shape

  # Define bounds
  bounds = [(0,shape[0]-(box_size-1)), (0,shape[1]-(box_size-1))]
  bounds = bounds * pixel_count

  # Constrain Population Size
  pop_size = max(1, pop_size // len(bounds))
  
  # Set Default Parameters for Callbacks
  p = partial(predict, image=image, detector=detector, box_size=box_size)
  c = partial(success, image=image, detector=detector, box_size=box_size)
  
  # Differential Evolution
  result = differential_evolution(p, bounds, maxiter=max_iter, popsize=pop_size,
                                  recombination=1, callback=c, polish=False)

  # Return Modification Array
  return result.x

# Load Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Get Cat Images
cat_idxs = [i for i, x in enumerate(y_train.flatten().tolist()) if x == 3]
cat_imgs = x_train[cat_idxs,:,:,:]
cat_imgs = cat_imgs.astype('float32') / 255.0

# Parameters
pixel_cnt = 1
max_iter = 50
pop_size = 50

# Load Detector
detector = load_model('trained_models/cifar10_detector_model.h5')

# Predict Images
preds = detector.predict(cat_imgs)

# Collect Data
for box_size in [4,1]:
  print("processing boxs: {}".format(box_size))
  data = []
  avg = 2.25
  rem = (len(cat_imgs) * avg) / 60.0
  
  for i, (image, pred) in enumerate(zip(cat_imgs, preds)):
    print('[{:.4f}, {:.4f}] processing image: {} of {} ({}, {})'.format(avg, rem, i, len(cat_imgs), box_size, pred[3]))
    if ((i % 500) == 0):
      draw_img = True

    if (pred[3] > 0.9):
      start = time.time()
      change = find_change(image, detector, pixel_cnt, max_iter=max_iter, pop_size=pop_size, box_size=box_size)
      end = time.time()
      avg = avg + ((end-start) - avg) / (i+1)
      rem = ((len(cat_imgs) - (i+1)) * avg) / 60.0
      new_img = modify(change, image, box_size)
      new_pred = detector.predict(np.reshape(new_img, (1,32,32,3)))[0]
      data.append({'image':image, 'orig_pred': pred, 'change': change, 'new_image': new_img, 'new_pred': new_pred})

      if (draw_img):
        draw_img = False
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].set_title("orig: {:.2f}".format(pred[3]))
        ax[0].axis('off')
        ax[1].imshow(new_img)
        ax[1].set_title('modified: {:.2f}'.format(new_pred[3]))
        ax[1].axis('off')
        fig.savefig("output/train_{}_{}.png".format(i, box_size))

  print('saving data')
  with open('output/cat_change_{}.p'.format(box_size), 'wb') as fd:
    pickle.dump(data, fd)
