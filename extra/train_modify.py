import pickle

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import load_model, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import SGD
from cifar10_resnet import resnet_v2

# Modify Function
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

# Block Size
box_size = 2
test_size = 200
epoch_size = 1

# Load Data
with open('output/cat_change_{}.p'.format(box_size), 'rb') as fd:
  data = pickle.load(fd)

# Get Training Data
cats = [x for x in data if any(x['new_pred'] > x['new_pred'][3])]
x_train = np.stack([x['image'] for x in cats])
y_train = np.stack([x['change'] for x in cats]) / (33.0 - box_size) # Normalize to 0 to 1


train_size = x_train.shape[0] - test_size
x_test = x_train[train_size:, :, :, :]
y_test = y_train[train_size:, :]
x_train = x_train[:train_size, :, :, :]
y_train = y_train[:train_size, :]

m_imgs = np.stack([x['new_image'] for x in cats])
m_imgs = m_imgs[:train_size, :, :, :]

## Build Model
#model = resnet_v2(input_shape=x_train[0, :, :, :].shape, depth=29, num_classes=2)
#model.compile(loss='squared_hinge', optimizer=Adam(lr=0.0000001),
#              metrics=['accuracy'])
#
## Train
#model.fit(x_train, y_train, batch_size=32, epochs=epoch_size,
#          validation_data=(x_test, y_test))

# My Model
img = Input(shape=x_train[0, :, :, :].shape)
x = Flatten()(img)
x = Dense(1000, activation='sigmoid')(x)
x = Dense(400, activation='sigmoid')(x)
x = Dense(2, activation='sigmoid')(x)
model = Model(img, x)
model.summary()

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.1),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=epoch_size, 
          validation_data=(x_test, y_test))

# Test
img_cnt = 5

# Select Test Images
idxs = np.random.randint(0, x_train.shape[0], img_cnt)
imgs = x_train[idxs, :, :, :]
mimgs = m_imgs[idxs, :, :, :]

# Build Detector
detector = load_model('trained_models/cifar10_detector_model.h5')

# Modify Images
fig, ax = plt.subplots(3, img_cnt)
for i in range(img_cnt):
  img = imgs[i, :, :, :]
  mimg = mimgs[i, :, :, :]
  pred = detector.predict(np.reshape(img, (1,32,32,3)))[0][3]
  mpred = detector.predict(np.reshape(mimg, (1,32,32,3)))[0][3]
  change = model.predict(np.reshape(img, (1,32,32,3)))[0] * 32.0
  new_img = modify(change, img, box_size)
  new_pred = detector.predict(np.reshape(new_img, (1,32,32,3)))[0][3]
  ax[0,i].imshow(img)
  ax[0,i].set_title('{:.2f}'.format(pred*100))
  ax[0,i].axis('off')
  ax[1,i].imshow(mimg)
  ax[1,i].set_title('{:.2f}'.format(mpred*100))
  ax[1,i].axis('off')
  ax[2,i].imshow(new_img)
  ax[2,i].set_title('{:.2f}'.format(new_pred*100))
  ax[2,i].axis('off')

fig.savefig("output/modify.png")