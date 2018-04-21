import os
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.models import load_model, Sequential, Model
from keras.layers import Input, Dense, Reshape, UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation
from keras.optimizers import RMSprop


def save_imgs(epoch, generator, detector):
  r, c = 5, 5
  noise = np.random.normal(0, 1, (r * c, 100))
  gen_imgs = generator.predict(noise)

  # Test One and Save Results
  y = detector.predict(gen_imgs)
  np.save('output/%d_img.npy' % epoch, gen_imgs)
  np.savetxt('output/%d_cat.txt' % epoch, y)

  # Rescale images 0 - 1
  gen_imgs = (1/2.5) * gen_imgs + 0.5

  fig, axs = plt.subplots(r, c)
  cnt = 0
  for i in range(r):
    for j in range(c):
      axs[i,j].imshow(gen_imgs[cnt, :,:,:])
      axs[i,j].axis('off')
      cnt += 1
  fig.savefig("output/%d.png" % epoch)
  plt.close()


def build_generator(noise_shape, img_shape):
  model = Sequential(name='gen')
  model.add(Dense(128 * 8 * 8, activation="relu", input_shape=noise_shape))
  model.add(Reshape((8, 8, 128)))
  model.add(BatchNormalization(momentum=0.8))
  model.add(UpSampling2D())
  model.add(Conv2D(128, kernel_size=4, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(UpSampling2D())
  model.add(Conv2D(64, kernel_size=4, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(3, kernel_size=4, padding="same"))
  model.add(Activation("tanh"))
  model.summary()

  noise = Input(shape=noise_shape)
  img = model(noise)

  return Model(noise, img, name='generator')


def train_gan(batch_size, epochs, model, generator, detector):
  for epoch in range(epochs):
    # Build Noise Vector
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Generator Wants Cats
    valid_y = np.ones((batch_size,))

    # Train the generator
    g_loss = model.train_on_batch(noise, valid_y )

    # Save Image
    if epoch % 1000 == 0:
      save_imgs(epoch, generator, detector)

      # Plot the progress
      print ("%5d [G loss: %f]" % (epoch, g_loss))


# Optimizer
optimizer = RMSprop(lr=0.000005)

# Load Detector
detector = load_model('trained_models/cat_detector_model.h5')
detector.trainable = False
#detector.summary()

# Build Generator
generator = build_generator((100,), (32,32,3))
generator.summary()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

# The generator takes noise as input and generated imgs
z = Input(shape=(100,))
img = generator(z)

# The valid takes generated images as input and determines validity
valid = detector(img)

# Buuild combined model
combined = Model(z, valid, name='combined')
combined.compile(loss='binary_crossentropy', optimizer=optimizer)
combined.summary()

# Make Output Directory
#os.mkdir('output')

# Train Generator
train_gan(32, 10001, combined, generator, detector)