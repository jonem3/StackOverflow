# from django.conf import settings
import math
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ...models import WeatherObservation, WeatherLocation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from datetime import datetime, timedelta
from tensorflow.keras.backend import square, mean

def loss_mse_warmup(y_true, y_pred):
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    mse = mean(square(y_true_slice - y_pred_slice))
    return mse



def batch_generator(batch_size, sequence_length, num_train, x_train_scaled, y_train_scaled, num_x_signals,
                    num_y_signals):
    # Generates random batches of training data
    while True:
        # Allocates a new array for each batch of input-signals
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocates a new array for each batch of output-signals
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # FIlls the batch with random sequences of data
        for i in range(batch_size):
            # Gets a random start-index which points somewhere into the training data
            idx = np.random.randint(num_train - sequence_length)

            # Copies the sequences of data starting at this index
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]

        yield (x_batch, y_batch)

def build_model():

    # CREATING THE RECURRENT NEURAL NETWORK
    model = Sequential()
    # Add a Gated Recurrent Network to the network, has 512 outputs for each time-step
    # Input shape requires the shape of its input, None means of arbitrary length
    model.add(GRU(units=512,
                  return_sequences=True,
                  input_shape=(None, num_x_signals,)))
    # As output signals are limited between 0 and 1 we must do the same for the output from the neural network
    model.add(Dense(num_y_signals, activation='sigmoid'))


    # LOSS FUNCTION

    global warmup_steps
    warmup_steps = 50

    # Defining the start Learning Rate we will be using
    optimizer = tf.keras.optimizers.RMSprop(lr=1e-10)
    # Compiles the model:
    model.compile(loss=loss_mse_warmup, optimizer=optimizer)
    model.summary()

    # Callback for writing checkpoints during training
    path_checkpoint = 'checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)

    # Callback for stopping optimization when performance worsens on the validation set
    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=5, verbose=1)

    # Callback for writing the TensorBoard log during training
    callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                       histogram_freq=0,
                                       write_graph=False)
    # Callback reduces learning rate for the optimizer if the validation-loss has not improved since the last epoch
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-4,
                                           patience=0,
                                           verbose=1)

    callbacks = [callback_early_stopping,
                 callback_checkpoint,
                 callback_tensorboard,
                 callback_reduce_lr]

    model.fit(x=generator,
              epochs=20,
              steps_per_epoch=100,
              validation_data=validation_data,
              callbacks=callbacks)

    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                            y=np.expand_dims(y_test_scaled, axis=0))
    print("loss (test-set):", result)

   
