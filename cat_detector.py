"""Trains a ResNet v2 on the CIFAR10 dataset, just looking for cats

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras.models import Model

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True,
                 conv_first=True):
  conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                padding='same', kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))

  x = inputs
  
  if conv_first:
    x = conv(x)
  
    if batch_normalization:
      x = BatchNormalization()(x)

    if activation is not None:
      x = Activation(activation)(x)

  else:
    if batch_normalization:
      x = BatchNormalization()(x)
    
    if activation is not None:
      x = Activation(activation)(x)
    
    x = conv(x)

  return x

def build_detector(input_shape, depth_scale):
  depth = 9 * depth_scale + 2
  num_filters_in = 16
  num_res_blocks = int((depth - 2) / 9)

  # Input Layer
  inputs = Input(shape=input_shape)

  # Build First Later
  x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

  # Middle Layers
  for stage in range(3):
    for res_block in range(num_res_blocks):
      activation = 'relu'
      batch_normalization = True
      strides = 1

      if stage == 0:
        num_filters_out = num_filters_in * 4
        if res_block == 0:  # first layer and first stage
          activation = None
          batch_normalization = False
      else:
        num_filters_out = num_filters_in * 2
        if res_block == 0:  # first layer but not first stage
            strides = 2    # downsample

        # bottleneck residual unit
        y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1,
                         strides=strides, activation=activation,
                         batch_normalization=batch_normalization,
                         conv_first=False)
        y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
        y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1,
                         conv_first=False)
        if res_block == 0:
          # linear projection residual shortcut connection to match
          # changed dims
          x = resnet_layer(inputs=x, num_filters=num_filters_out,
                           kernel_size=1, strides=strides, activation=None,
                           batch_normalization=False)

        x = keras.layers.add([x, y])

    num_filters_in = num_filters_out

  # Add Classifier
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = AveragePooling2D(pool_size=8)(x)
  y = Flatten()(x)
  outputs = Dense(2, activation='relu', kernel_initializer='he_normal')(y)

  # Create Model
  model = Model(inputs=inputs, outputs=outputs)

  return model
