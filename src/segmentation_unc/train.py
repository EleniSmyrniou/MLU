import math
import os
import numpy as np

from dataset import get_train_test_validation_data
from model import get_model

from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint


def train(epochs, batch_size, initial_epoch, directory, test_percentage, validation_percentage):
    """Trains a model."""
    print('loading data...')
    # Loads or creates training data.
    train_input, train_target, test_input, test_target, validation_input, validation_target = \
        get_train_test_validation_data(directory, test_percentage, validation_percentage)
    # define input shape
    input_shape = train_input[0].shape
    print('getting model...')
    # Loads or creates model.
    model, checkpoint_path = get_model(input_shape)
    #
    # # Sets callbacks.
    checkpointer = ModelCheckpoint(checkpoint_path, verbose=1,
                                    save_weights_only=True, save_best_only=True)

    print('fitting model...')
    # # Trains model.
    model.fit(train_input, train_target, batch_size, epochs,
               initial_epoch=initial_epoch)
    print('evaluating model...')
    # # Evaluates model for each test input.
    mc_samples = 100
    sigmoids = np.zeros((mc_samples,) + test_target[0].shape)
    # loop according to batch size
    for i in range(0, len(test_input), batch_size):
        for j in range(mc_samples):
            # get batch
            batch = test_input[i:i + batch_size]
            # predict
            batch_sigmoids = model.predict(batch)
            # add to sigmoids
            sigmoids[j, i:i + batch_size] = batch_sigmoids
        # get batch
        batch = test_input[i:i + batch_size]
        # predict
        batch_sigmoids = model.predict(batch)
        # add to sigmoids
        sigmoids[i:i + batch_size] = batch_sigmoids
    # Calculates prediction.
    pred = np.mean(sigmoids, axis=0)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.

    print('done')

