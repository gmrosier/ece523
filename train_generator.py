import os
import numpy as np
from matplotlib import pyplot as plt
import keras
import keras.backend as K
from keras.models import load_model, Sequential, Model
from keras.layers import Input, Dense, Reshape, UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, Add, Multiply
from keras.optimizers import RMSprop
from keras.datasets import cifar10

def save_imgs(epoch, noise, images, scale, batch_size, generator, detector):
  r, c = 5, 5
  idxs = np.random.randint(0, batch_size, (r * c))
  n = noise[idxs, :]
  i = images[idxs, :, :, :]
  s = scale[idxs, :, :, :]
  #gen_imgs = generator.predict([n, s, i])
  gen_imgs = generator.predict(n)

  # Make Output Dir
  os.makedirs('output', exist_ok=True)

  # Test One and Save Results
  y = detector.predict(gen_imgs) * 100
  np.save('output/%d_img.npy' % epoch, gen_imgs)
  np.savetxt('output/%d_cat.txt' % epoch, y)

  # Rescale images 0 - 1
  gen_imgs -= gen_imgs.min()
  gen_imgs /= gen_imgs.max()

  fig, axs = plt.subplots(r, c)
  cnt = 0
  for i in range(r):
    for j in range(c):
      axs[i,j].imshow(gen_imgs[cnt, :,:,:])
      axs[i,j].axis('off')
      cnt += 1
  fig.savefig("output/%d.png" % epoch)
  plt.close()


def build_generator(noise_shape, image_shape):
  noise = Input(shape=noise_shape)
  #img = Input(shape=image_shape)
  #scale = Input(shape=image_shape)
  x = Dense(128 * 8 * 8, activation="relu", input_shape=noise_shape)(noise)
  x = Reshape((8, 8, 128))(x)
  x = BatchNormalization(momentum=0.8)(x)
  x = UpSampling2D()(x)
  x = Conv2D(128, kernel_size=4, padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(momentum=0.8)(x)
  x = UpSampling2D()(x)
  x = Conv2D(64, kernel_size=4, padding="same")(x)
  x = Activation("relu")(x)
  x = BatchNormalization(momentum=0.8)(x)
  x = Conv2D(3, kernel_size=4, padding="same")(x)
  x = Activation("sigmoid")(x)
  #x = Multiply()([x, scale])
  #x = Add()([x, img])

  #return Model([noise, scale, img], x, name='generator')
  return Model(noise, x, name='generator')


def train_gan(batch_size, epochs, model, generator, detector):

  # Load Data
  _, (x_test, y_test) = cifar10.load_data()

  # Load One Cat Image
  cat_idxs = [i for i, x in enumerate(y_test.flatten().tolist()) if x == 3]
  cat_imgs = x_test[cat_idxs,:,:,:]
  cat_imgs = cat_imgs.astype('float32') / 255.0
  cat_imgs = cat_imgs[0:batch_size, :, :, :]

  e_scale = a = np.linspace(0.1,1,10001)**6

  for epoch in range(epochs):
    # Build Noise Vector
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generator Wants Cats
    valid_y = np.ones((batch_size,)) * 3
    y_valid = keras.utils.to_categorical(valid_y, 10)

    # Create Scale
    scale = np.ones(cat_imgs.shape) * e_scale[epoch]

    # Scale Image
    imgs = cat_imgs * (1 - scale)

    # Train the generator
    metrics = model.train_on_batch(noise, y_valid)
    #metrics = model.train_on_batch([noise, scale, imgs], valid_y)

    # Save Image
    if epoch % 1000 == 0:
      save_imgs(epoch, noise, imgs, scale, batch_size, generator, detector)

      # Plot the progress
      print ("%5d [G loss: %f, accuracy: %f]" % (epoch, metrics[0], metrics[1]))

# Optimizer
optimizer = RMSprop(lr=0.0001)

# Load Detector
detector = load_model('trained_models/cifar10_detector_model.h5')
detector.summary()


# Build Generator
image_shape = (32,32,3)
generator = build_generator((100,), image_shape)
generator.summary()
generator.compile(loss='categorical_crossentropy', optimizer=optimizer)

# The generator takes noise as input and generated imgs
z = Input(shape=(100,))
oimg = Input(shape=image_shape)
scl = Input(shape=image_shape)
#img = generator([z, scl, oimg])
img = generator(z)

# The valid takes generated images as input and determines validity
valid = detector(img)

# Buuild combined model
#combined = Model([z, scl, oimg], valid, name='combined')
combined = Model(z, valid, name='combined')

# Freeze Detector Layers
for layer in detector.layers:
    layer.trainable = False

# Compile Model
combined.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
combined.summary()

# Train Generator
train_gan(32, 10001, combined, generator, detector)

# Save Generator
generator.save('trained_models/cat_generator.hd5')