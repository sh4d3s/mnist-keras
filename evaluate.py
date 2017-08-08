from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import mnist_data
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')

batch_size = 128
num_classes = 10
epochs = 12

train_total_data, train_size, validation_data, validation_labels, x_test, y_test = mnist_data.prepare_MNIST_data(True)
# input image dimensions
img_rows, img_cols = 28, 28

x_train = train_total_data[:, :-10]
y_train = train_total_data[:, -10:]

x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)
validation_data = validation_data.reshape(validation_data.shape[0],img_rows, img_cols,1)
input_shape = (img_rows, img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
validation_data = validation_data.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape, 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('MNIST.hdf5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
