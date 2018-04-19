import keras
from keras.models import load_model
from keras.datasets import cifar10

# Load Model
model = load_model('trained_models/cat_detector_model.h5')

# Load Data
_, (x_test, y_test) = cifar10.load_data()

# Convert from uint8 and normalize
x_test = x_test.astype('float32') / 255

# Convert to Binary (Cat / Not Cat)
y_test[y_test != 3] = 0
y_test[y_test == 3] = 1
y_test = keras.utils.to_categorical(y_test, 2)

# Score Training
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])