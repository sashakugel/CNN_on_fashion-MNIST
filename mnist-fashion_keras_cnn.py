import numpy as np
np.random.seed(1983)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.datasets import fashion_mnist

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()						#import data 

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)									#specify image coloring dimension: 1 for b&w, 3 for rgb
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')														#transposing from uint8 to float32
X_test = X_test.astype('float32')
X_train /= 255																			#normalizing to values 0-255 (??)
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)											#from 1-dim class array to 10-class matrix [mendatory for the model later)
Y_test = np_utils.to_categorical(y_test, 10)

# 7. Define model architecture
model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))					#1st convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))										#2st convolutional layer
model.add(MaxPooling2D(pool_size=(2,2)))												#Pooling
model.add(Dropout(0.25))
 
model.add(Flatten())																	#Flatten
model.add(Dense(128, activation='relu'))												#ReLu
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))												#softmax
 REALL
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',															#??
              metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)
 
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print("score:", score)
