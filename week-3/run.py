#!/usr/bin/python


import numpy as np
from keras.layers import Dense
from keras.models import Sequential


model = Sequential()
model.add(Dense(1, activation='linear', input_shape=(1,)))

model.summary()

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

x_train = np.array(range(1, 10000, 1))
y_train = np.array(range(2, 20000, 2))
x_test = np.array(range(3, 300, 3))
y_test = np.array(range(6, 600, 6))

model.fit(x_train, y_train, epochs=100)

print "Evaluation"
print model.evaluate(x_test, y_test)
