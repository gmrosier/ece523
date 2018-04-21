import os
import numpy as np

import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10

from cat_detector import build_detector

def train_detector(x_train, y_train, x_test, y_test, batch_size, epochs):

  # Learning Rate Scheduler
  def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

  # Build Model
  model = build_detector(input_shape, 3)

  # Train
  model.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=lr_schedule(0)),
                metrics=['accuracy'])
  model.summary()

  # Prepare Saved Model Dir
  save_dir = os.path.join(os.getcwd(), 'trained_models')
  model_name = 'cat_detector_model.{epoch:03d}.h5'
  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
  filepath = os.path.join(save_dir, model_name)

  # Prepare Callbacks
  checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1,
                               save_best_only=True)
  lr_scheduler = LearningRateScheduler(lr_schedule)
  lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                 min_lr=0.5e-6)
  callbacks = [checkpoint, lr_reducer, lr_scheduler]

  # Run Training
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)

  return model

# Load Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Get Input Size
input_shape = x_train.shape[1:]

# Convert from uint8 and normalize
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Subtract Pixel Mean to Improve Accuracy
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

# Convert to Binary (Cat / Not Cat)
y_train[y_train != 3] = 0
y_train[y_train == 3] = 1
y_test[y_test != 3] = 0
y_test[y_test == 3] = 1

# Train Model
model = train_detector(x_train, y_train, x_test, y_test, 32, 200)

# Score Training
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
