from os import environ

import json

from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import BatchNormalization
from keras.layers import Conv2D

environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda0,floatX=float32,optimizer=None'
environ['KERAS_BACKEND'] = 'theano'

# ------------------------ Model from nvidia ------------------------

model = Sequential()

model.add(BatchNormalization(epsilon=0.001, axis=1, input_shape=(66, 200, 3)))

model.add(Conv2D(24, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(36, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(48, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Conv2D(64, (3,3), padding='valid', activation='relu', strides=(1,1)))
model.add(Conv2D(64, (3,3), padding='valid', activation='relu', strides=(1,1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.summary()

# ------------------------ Saving the model as json file ------------------------

with open('model.json', 'w') as jsonfile:
    jsonfile.write(json.dumps(json.loads(model.to_json()), indent=2))
