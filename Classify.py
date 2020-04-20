import gc
import os, re
from os import rename
import random
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.api.keras import models, layers, optimizers

train_directory = 'Dataset'
test_directory = 'Test'

train_poledance = []
for i in os.listdir(train_directory):
    if 'Pole' in i:
        train_poledance.append('Dataset/{}'.format(i))
train_yoga = []
for i in os.listdir(train_directory):
    if 'Yoga' in i:
        train_yoga.append('Dataset/{}'.format(i))


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


train_images = train_poledance[:500] + train_yoga[0:500]
random.shuffle(train_images)
# test_images = [test_directory+i for i in os.listdir(test_directory)]
test_images = []
for i in os.listdir(test_directory):
    test_images.append('Test/{}'.format(i))

print(len(test_images))
test_images.sort(key=natural_keys)

# clear images from this variables
del train_yoga
del train_poledance
gc.collect()

# resize the image
nrows = 150
ncolumns = 150
# this is for RGB
channels = 3


# function to read and process the image
def process_image(list_of_images):
    X = []
    y = []
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
    for img in list_of_images:
        if 'Pole' in img:
            y.append(1)
        elif 'Yoga' in img:
            y.append(0)
    return X, y


# Generate x_train and y_train
X, y = process_image(train_images)


# Generate X_test and Y_test
def process_test_image(list_of_images):
    x_test = []
    y_test = []
    for imag in test_images:
        x_test.append(cv2.resize(cv2.imread(imag), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
    return x_test, y_test


x_test, y_test = process_test_image(test_images)

# split the data into train and test set
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=1)
len_train = len(X_train)
len_val = len(X_val)
batch_size = 25

# design model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(nrows, ncolumns, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

# Normalization
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# prepare generators for training and validation sets
train_gen = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)

validation_gen = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)

# Start training the model
history = model.fit_generator(train_gen,
                              steps_per_epoch=len_train // batch_size,
                              epochs=50,
                              validation_data=validation_gen,
                              validation_steps=len_val // batch_size
                              )
model.save_weights('best_model_weights_6.h5')
model.save('classification_model_6.h5')

# plot accuracy

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Normalization
test_datageneration = ImageDataGenerator(rescale=1. / 255)

# prepare test set and predict value
test_generation = val_datagen.flow(np.array(x_test), batch_size=batch_size)
prediction = model.predict(test_generation, verbose=1)

# Generate csv file for reference
count = range(1, len(test_images) + 1)
sol = pd.DataFrame({"id": count, "Prediction": list(prediction)})
cols = ['Prediction']
for col in cols:
    sol[col] = sol[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
    sol.to_csv('Classification.csv', index=False)

