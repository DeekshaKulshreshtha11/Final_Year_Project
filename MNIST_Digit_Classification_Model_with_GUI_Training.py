import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def plot_input_img(i):
    plt.imshow(X_train[i], cmap='binary')
    plt.title(y_train[i])
    plt.show()

#Pre-Processing the data
#Normalizing the image to the scale of [0-1] range
X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

#ReShape / expand the dimension of the image to (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Converting classes to one Hot Vector
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
# print(y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))
model.summary()

model.compile(optimizer='adam', loss= keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

#earlyStopping
es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)

#ModelCheckPoint
mc = ModelCheckpoint("/.bestmodel.h5", monitor="val_acc", verbose=1, save_best_only=True)
cb = [es, mc]

# Training the model
trained_model = model.fit(X_train, y_train, epochs=50, validation_split=0.3)
model.save('../MNIST_Digit_Classification_Model_with_GUI')