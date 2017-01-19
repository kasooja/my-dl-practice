from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np


seed = 7
np.random.seed(seed)


def baseline_model():
    model = Sequential()
    model.add(Dense(512, input_dim=num_pixels, init='normal', activation='relu'))
    #model.add(Dense(512, input_shape=(784,)))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

mnistData = mnist.load_data()
X_train, y_train = mnistData[0]
X_test, y_test = mnistData[1]

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train /= 255
X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = baseline_model()
model.fit(X_train, y_train, validation_split=0.1,
          nb_epoch=5, batch_size=128, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=2)
print(model.metrics_names)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

