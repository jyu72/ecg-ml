'''ecg_cnn.py

'''

from __future__ import print_function
import os
import ecg_data
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers import Dense, Flatten, Merge, Concatenate


RECORD_LENGTH = 650000  # each ECG record contains 650000 samples total


# --------------------------- Training CNN Models ---------------------------- #


# Train a 2D CNN model on selected ECG data
def train_2d(record_numbers, sampfrom=0, sampto=RECORD_LENGTH, channels=[0,1], 
             beat_types=['V'], window_radius=24, graph=False, save=False, 
             file_name='unnamed'):
    """
    Train a 2D CNN model on selected ECG data. 

    The channels (i.e. ECG leads) will be processed together with 
    convolutional filters and then learned.

    Parameters
    ----------
    record_numbers : list of int
        List of ECG records to be used for training.

    sampfrom : int, optional
        First sample number read.

    sampto : int, optional
        Last sample number read.

    channels : list of int, optional
        The ECG channels (must be 2) that will be used for training.

    beat_types : list of string, optional
        The ECG beat types that the model will be trained on.

    window_radius : int, optional
        Number of samples that are used on each side of a selected sample. The
        window will have length (1 + 2 * window_radius).

    graph : Boolean, optional
        Whether the performance of the model during training is graphed.

    save : Boolean, optional
        Whether the model is saved to a file.

    file_name : string, optional
        The file name that the model will be saved with extension .h5.

    Returns
    -------
    model : keras.models.Sequential
        The trained 2D CNN model.

    Examples
    --------
    >>> model = ecg_cnn.train_2d([119], sampto=10800, beat_types=['V'], 
                        window_radius=24, graph=True, save=True, file_name='experiment')

    """

    # CNN parameters
    batch_size = 20
    epochs = 20
    pool_size = 2
    kernel_size = (1, 5)
    strides = (1, 1)

    # Additional class is for all "other" beats
    num_classes = len(beat_types) + 1

    # Load train, validation, and test data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = ecg_data.load_datasets_windows_2d(record_numbers, sampfrom, sampto, beat_types, window_radius)

    # Input dimensions
    img_y = 2 * window_radius + 1
    img_x = 2
    input_shape = (img_x, img_y, 1)
    n_train_examples = x_train.shape[0]
    n_val_examples = x_val.shape[0]
    n_test_examples = x_test.shape[0]

    # Reshape the data
    x_train = x_train.reshape(n_train_examples, img_x, img_y, 1)
    x_val = x_val.reshape(n_val_examples, img_x, img_y, 1)
    x_test = x_test.reshape(n_test_examples, img_x, img_y, 1)

    # Create CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=kernel_size, strides=strides, padding='same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=kernel_size, strides=strides, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(32, kernel_size = kernel_size, strides=strides, padding='same', 
                    activation='relu'))
    model.add(Conv2D(32, kernel_size = kernel_size, strides=strides, padding='same', 
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Fit (train) model to train data and utilize validation data
    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[history])

    # Evaluate performance of model on test data
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # Graph training performance over epochs
    if graph:
        plt.plot(range(1, 1 + epochs), history.acc)
        plt.xlim(0)
        plt.xticks(range(1, 1 + epochs))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

    # Save model to file
    if save:

        if not os.path.exists('models'):
            os.makedirs('models')

        model.save('models/' + file_name + '.h5')
        print('Saved model to models/' + file_name + '.h5')

    return model


# ---------------------------- Making Predictions ---------------------------- #


# Use trained CNN model to make predictions on unannotated data
def predict_2d(model, samples, windows, beat_types=['V'], window_radius=24):

    # Additional class is for all "other" beats
    classes = ['-'] + beat_types

    # Format the data
    img_y = 2 * window_radius + 1
    img_x = 2
    input_shape = (img_x, img_y, 1)
    n = windows.shape[0]
    windows = windows.reshape(n, img_x, img_y, 1)

    # Make predictions
    y_prediction = model.predict(windows)

    # Convert multi-class probabilities into class predictions
    one_hot_encoded = np.argmax(y_prediction, axis=1)
    y_labels = [classes[i] for i in one_hot_encoded]

    return y_labels

