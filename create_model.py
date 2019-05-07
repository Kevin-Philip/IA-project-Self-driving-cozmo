import os
import json
import keras.models as models

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda0,floatX=float32,optimizer=None'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization
from keras.layers import Conv2D

imgSize = (66, 200, 3) # h, w, channels

# We used the model made by Nvidia: https://arxiv.org/pdf/1604.07316.pdf

model = Sequential()

model.add(BatchNormalization(epsilon=0.001, axis=1, input_shape=imgSize))

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

# Save model to JSON
with open('autopilot_basic_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))