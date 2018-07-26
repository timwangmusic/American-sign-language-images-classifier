# Neural network for classifying american sign language letters
import signdata
import numpy as np
from keras import models
from keras import layers
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.team_name = "qualcoder"
config.loss = "categorical_crossentropy"
config.optimizer = "rmsprop"
config.epochs = 5

if (config.team_name == 'default'):
    raise ValueError("Please set config.team_name to be your team name")

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

# define image data generator
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)


# reshape to 4 dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# get number of classes
num_classes = y_train.shape[1]

# model building
network = models.Sequential()
network.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1,)))
network.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Dropout(0.3))
network.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Conv2D(128, (2,2), padding='same', activation='relu'))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Dropout(0.3))
network.add(layers.Flatten())
network.add(layers.Dense(1500, activation='relu'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(num_classes, activation='softmax'))

# network.summary()

# model compilation
import keras.optimizers
network.compile(optimizer='adam',
               loss = 'categorical_crossentropy',
               metrics=['accuracy'])

# keras.optimizers.RMSprop(lr=5e-4)
# Fit the model
history_callback = network.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=10, validation_data=(X_test, y_test),
                        callbacks=[WandbCallback(data_type="image", labels=signdata.letters)],
                        verbose=1)

# network.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
#                     callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
