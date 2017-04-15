import csv
import cv2
import numpy as np
from sklearn.utils import shuffle


STEERING_CORRECTION = .2
INDEX_OF_STEERING_ANGLE = 3
IM_NORMAL = 'normal'
IM_FLIPPED = 'flipped'

"""
Read training data file names in
"""
lines = []
for dir_name in ['data']:
    with open(dir_name +'/driving_log.csv') as csvfile:
        reader  = csv.reader(csvfile)
        for line in reader:
            for i in range(3):
                measurement = float(line[INDEX_OF_STEERING_ANGLE])
                if i == 1:
                    measurement += STEERING_CORRECTION
                elif i == 2:
                    measurement -= STEERING_CORRECTION
                #Append the image normally, and the image with a tag for flipping
                lines.append((line[i],measurement,IM_NORMAL))
                lines.append((line[i],measurement,IM_FLIPPED))

from sklearn.model_selection import train_test_split
lines = shuffle(lines)

# Split training data into training and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def read_img(source_path):
    """
    Given a filename, returns imread image
    """
    source_path = source_path 
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    return cv2.imread(current_path)

def generator(samples, batch_size=32):
    """
    Generator that yields 32 images at a time for training.
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Read Image
                image = read_img(batch_sample[0])
                measurement = batch_sample[1]

                # Augment Data - If image should be flipped, flip it.
                if batch_sample[2] == IM_FLIPPED:
                    images.append(cv2.flip(image,1))
                    measurements.append(measurement*-1.0)
                else:
                    images.append(image)
                    measurements.append(measurement)

            batch_images = np.array(images)
            batch_angles = np.array(measurements)
            yield batch_images, batch_angles

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model

def le_net():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - .5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    return model

def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - .5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    return model


model = nvidia()
#model = le_net()
#history_object = model.fit(X_train,y_train,validation_split=.2,shuffle=True,nb_epoch=3)
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     validation_data = validation_generator,
                                     nb_val_samples = len(validation_samples),
                                     nb_epoch=5)
model.save('model.h5')
