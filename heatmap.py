import pickle

import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import keras.backend as K
from keras.models import load_model


# Params
box_size = 2
cat_idx = 11

# Load Data
with open('trained_models/cat_change_{}.p'.format(box_size), 'rb') as fd:
  data = pickle.load(fd)

# Get Training Data
cats = [x for x in data if any(x['new_pred'] > x['new_pred'][3])]
x_train = np.stack([x['image'] for x in cats])
new_imgs = np.stack([x['new_image'] for x in cats])
new_pred = np.stack([x['new_pred'] for x in cats])

# Load Detector
detector = load_model('trained_models/cifar10_detector_model.h5')
detector.summary()

# Build Heatmap Model
last_conv = None
for layer in detector.layers[::-1]:
  if layer.name.startswith('conv'):
    last_conv = layer
    break

if last_conv is not None:
  class_weights = detector.layers[-1].get_weights()[0]
  get_output = K.function([detector.layers[0].input],
                          [last_conv.output, detector.layers[-1].output])

  img = np.expand_dims(x_train[cat_idx, :, :, :], axis=0)
  [conv_outputs, predictions] = get_output([img])
  conv_outputs = conv_outputs[0, :, :, :]
  
  #Create the class activation map.
  cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[:2])
  for i, w in enumerate(class_weights[:, 1]):
          cam += w * conv_outputs[:, :, i]

  cam /= np.max(np.abs(cam))
  cam *= -1.0
  cam[np.where(cam < 0.2)] = 0
  cam = resize(cam, img.shape[1:3])

  heatmap = cm.hot(cam)[:,:,:3]
  oimg = np.squeeze(img, axis=0)
  nimg = new_imgs[cat_idx, :, :, :]
  himg = heatmap * 0.35 + oimg * 0.65
  hnimg = heatmap * 0.1 + nimg + 0.9
  max_idx = np.unravel_index(cam.argmax(), cam.shape)
  mimg = np.copy(oimg)
  mimg[max_idx[0], max_idx[1], :] = [0, 0, 1.0]
  mimg[max_idx[0], max_idx[1]+1, :] = [0, 0, 1.0]
  mimg[max_idx[0]+1, max_idx[1], :] = [0, 0, 1.0]
  mimg[max_idx[0]+1, max_idx[1]+1, :] = [0, 0, 1.0]

  pred = detector.predict(np.expand_dims(mimg, axis=0))
  fig, ax = plt.subplots(2, 3)
  ax[0,0].imshow(oimg)
  ax[0,0].set_title("orig: {:.2f}".format(predictions[0][3]))
  ax[0,1].axis('off')
  ax[0,1].imshow(heatmap)
  ax[0,1].set_title("heatmap")
  ax[0,0].axis('off')
  ax[0,2].imshow(himg)
  ax[0,2].set_title("heatmap overlay")
  ax[0,2].axis('off')
  ax[1,0].imshow(nimg)
  ax[1,0].set_title("mod: {:.2f}".format(new_pred[cat_idx][3]))
  ax[1,0].axis('off')
  ax[1,1].imshow(mimg)
  ax[1,1].set_title("heat: {:.2f}".format(pred[0][3]))
  ax[1,1].axis('off')
  #ax[1,2].imshow(hnimg)
  ax[1,2].axis('off')
  
  plt.show()
