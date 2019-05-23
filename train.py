from os import environ

environ['KERAS_BACKEND'] = 'theano'
environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda0,floatX=float32,optimizer=None'

import keras.models as models
from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np
import glob
import record

import matplotlib.pyplot as plt

# ------------------------ Initialisation ------------------------

# Retrieve images sizes from recording
imgSize = record.imgSize

# ------------------------ Load training data ------------------------

# Create training data arrays
images = np.zeros((0, imgSize[0], imgSize[1], imgSize[2]), dtype=np.float16)
directions = np.zeros(0, dtype=np.float32)

# Retrieve files names
images_files = sorted(glob.glob('data_train/*-images.npz'))
directions_files = sorted(glob.glob('data_train/*-directions.npz'))

# Load training data into arrays
for imgfile, dirfile in zip(images_files, directions_files):
    images = np.append(images, np.load(imgfile)['images'], axis=0)
    directions = np.append(directions, np.load(dirfile)['directions'], axis=0)

print(f'Have {images.shape[0]} training images')

# ------------------------ Shuffle images and directions ------------------------

index = np.arange(0,images.shape[0])
index = np.random.permutation(index)
images = images[index,:,:,:]
directions = directions[index]

# ------------------------ Traning phase ------------------------

# Load model
model = Sequential()
with open('model.json') as model_file:
    model = models.model_from_json(model_file.read())

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.0001))

epochs = 25
batch_size = 64

# Train model
model.fit(images, directions, 
	batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

# Save weights
model.save_weights('weights/model_weights.hdf5')