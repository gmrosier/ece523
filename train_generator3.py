# This is build from the following paper
#
# One pixel attack for fooling deep neural networks
# Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai
# http://arxiv.org/abs/1710.08864


from functools import partial
import numpy as np
from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Lambda
from keras.datasets import cifar10

def modify(x, image):
  pixels = len(x)//5
  shape = (pixels, 5)
  x = np.reshape(x, shape)
  
  # Copy Image
  image = np.copy(image)

  # Modify The Pixes
  for i in range(pixels):
    x_idx, y_idx, *rgb = x[i, :]
    x_idx = int(np.floor(x_idx))
    y_idx = int(np.floor(y_idx))
    image[x_idx, y_idx] = rgb

  return image


def predict(x, image, detector):
  # Modifiy the specified pixels
  mimg = modify(x, image)
  
  # Test New Image
  pred = detector.predict(np.reshape(mimg, (1,32,32,3)))[0][0]

  # Return Prediction
  return pred


def success(x, convergence=0.1, image=None, detector=None):
  # Modifiy the specified pixels
  mimg = modify(x, img)
  pred = detector.predict(np.reshape(mimg, (1,32,32,3)))[0][0]
  
  # Return if image is no longer a cat
  return (pred < 0.4) 


def modify_image(image, detector, pixel_count=1, max_iter=10, popsize=500):

  # Get Image Shape
  shape = image.shape

  # Define bounds
  bounds = [(0,shape[0]), (0,shape[1])]
  for _ in range(shape[2]):
    bounds.append((0,1))
  bounds = bounds * pixel_count

  # Constrain Population Size
  popsize = max(1, popsize // len(bounds))
  
  # Set Default Parameters for Callbacks
  p = partial(predict, image=image, detector=detector)
  c = partial(success, image=image, detector=detector)
  
  # Differential Evolution
  result = differential_evolution(p, bounds, maxiter=max_iter, popsize=popsize,
                                  callback=c, polish=False, disp=True, atol=-1)

  # Return Modified Image
  return modify(result.x, image)


def build_detector():
  def cat_only(x):
    z = np.zeros((10,1))
    z[3] = 1
    y = K.constant(z)
    return K.dot(x, y)

  # Load Detector
  base_model = load_model('trained_models/cifar10_detector_model.h5')
  y = Lambda(cat_only)(base_model.output)
  return Model(base_model.input, y, name='detector')



# Load Data
_, (x_test, y_test) = cifar10.load_data()

# Get Cat Images
cat_idxs = [i for i, x in enumerate(y_test.flatten().tolist()) if x == 3]
cat_imgs = x_test[cat_idxs,:,:,:]
cat_imgs = cat_imgs.astype('float32') / 255.0

# Parameters
img_cnt = 5
pixel_cnt = 1
max_iter = 50

# Select Test Images
idxs = np.random.randint(0, cat_imgs.shape[0], img_cnt)
imgs = cat_imgs[idxs, :, :, :]

# Build Detector
detector = build_detector()

# Modify Images
fig, ax = plt.subplots(2, img_cnt)
for i in range(img_cnt):
  print("Modifing Cat: {}".format(i))
  img = imgs[i, :, :, :]
  pred = detector.predict(np.reshape(img, (1,32,32,3)))[0][0]
  new_img = modify_image(img, detector, pixel_cnt, max_iter=max_iter)
  new_pred = detector.predict(np.reshape(new_img, (1,32,32,3)))[0][0]
  ax[0,i].imshow(img)
  ax[0,i].set_title('{:.2f}'.format(pred*100))
  ax[0,i].axis('off')
  ax[1,i].imshow(new_img)
  ax[1,i].set_title('{:.2f}'.format(new_pred*100))
  ax[1,i].axis('off')

fig.savefig("output/modify.png")
