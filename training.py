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

# Takes data from databases and turns into a Pandas table for interaction with Tensorflow
def build_tables():
    timestamp = []
    locations = []

    # Searches through the WeatherLocation table to find all the names, being sorted by id
    for location in WeatherLocation.objects.filter().order_by("id"):
        if location.name in locations:
            continue
        locations.append(location.name)
        for obs in WeatherObservation.objects.filter(weather_location__id=location.id).order_by("time_stamp"):
            if timestamp is not None and obs.time_stamp not in timestamp:
                timestamp.append(obs.time_stamp)
    mux = pd.MultiIndex.from_product([locations, ['wind gust',
                                                  'temperature',
                                                  'wind speed',
                                                  'pressure',
                                                  'dew point',
                                                  'screen relative humidity',
                                                  'weather type',
                                                  'max uv',
                                                  'precipitation probability']])
    df = pd.DataFrame(columns=mux, index=timestamp)

    for location in WeatherLocation.objects.all().order_by("id"):
        for obs in WeatherObservation.objects.filter(weather_location__id=location.id).order_by("time_stamp"):
            if timestamp is not None:
                df.xs(obs.time_stamp)[location.name, 'wind gust'] = obs.wind_gust
                df.xs(obs.time_stamp)[location.name, 'temperature'] = obs.temperature
                df.xs(obs.time_stamp)[location.name, 'wind speed'] = obs.wind_speed
                df.xs(obs.time_stamp)[location.name, 'pressure'] = obs.pressure
                # df.xs(obs.time_stamp)[location.name, 'pressure tendency'] = obs.pressure_tendency
                df.xs(obs.time_stamp)[location.name, 'dew point'] = obs.dew_point
                df.xs(obs.time_stamp)[location.name, 'screen relative humidity'] = obs.screen_relative_humidity
                # df.xs(obs.time_stamp)[location.name, 'visibility'] = obs.visibility
                # df.xs(obs.time_stamp)[location.name, 'wind direction'] = obs.wind_direction
                df.xs(obs.time_stamp)[location.name, 'weather type'] = obs.weather_type
                df.xs(obs.time_stamp)[location.name, 'max uv'] = obs.max_uv
                df.xs(obs.time_stamp)[location.name, 'precipitation probability'] = obs.precipitation_probability

    return df


def loss_mse_warmup(y_true, y_pred):
    """
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    loss = tf.losses.mse(y_true_slice, y_pred_slice)

    loss_mean = tf.reduce_mean(loss)

    return loss_mean

    :param y_true:
    :param y_pred:
    :return:
    """
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

def plot_comparison(start_idx, length, train, x_train_scaled, y_train, y_test, y_scaler, model, x_test_scaled):
    """
    Plot the predicted and true output-signals.

    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """

    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test

    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]

        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15, 5))

        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')

        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()




def build_model():
    print("tensorflow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)
    print("Pandas version:", pd.__version__)
    df = build_tables()
    df.replace(to_replace=[None], value=np.nan, inplace=True)
    df = df.dropna(axis='columns')
    print(df)
    target_location = 'Wittering'

    global target_names
    target_names = ['temperature', 'wind speed']

    # Setting the range for when we want to predict the data (Presently set to 24 hours)
    shift_days = 1
    shift_steps = shift_days * 24
    df_targets = df[target_location][target_names].shift(-shift_steps)

    df['Wittering']['temperature'].plot()
    plt.show()

    # Converting the data from a Pandas table to a Numpy array as they can be entered into the Neural Network:

    # Input signals:
    x_data = df.values[0:-shift_steps]
    print("Shape:", x_data.shape)

    # Output signals
    y_data = df_targets.values[:-shift_steps]
    print(type(y_data))
    print("Shape:", y_data.shape)

    # Number of observations
    num_data = len(x_data)
    print("num_data =", num_data)
    print("num_data[0]=", x_data[0])
    # Defining the fraction of the data set to be used for the training set
    train_split = 0.9
    num_train = int(train_split * num_data)

    num_test = num_data - num_train

    # Input signals for training and test sets
    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]
    print("Input signals:", len(x_train)+len(x_test))

    # Output signals for training and test sets
    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]
    print("Output signals:", len(y_train) + len(y_test))

    # Number of input signals
    num_x_signals = x_data.shape[1]

    # Number of output signals
    num_y_signals = y_data.shape[1]

    # Outputs the range of the pre-scaled data
    print("Min:", np.min(x_train))
    print("Max:", np.max(x_train))

    # Creates a scalar object
    x_scaler = MinMaxScaler()
    # Detects the range of values from the training-data and scale the training-data
    x_train_scaled = x_scaler.fit_transform(x_train)

    # Data has now been scaled to between 0 and 1 (Not taking into account a small rounding error)
    print("Min:", np.min(x_train_scaled))
    print("Max:", np.max(x_train_scaled))

    # Same scaler object is used for the input-signals in the test set
    x_test_scaled = x_scaler.transform(x_test)

    # Repeats the same process for the output signals
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    print("Min:", np.min(y_train_scaled))
    print("Max:", np.max(y_train_scaled))
    y_test_scaled = y_scaler.transform(y_test)

    # Outputs the array-shapes of the input and output data
    print(x_train_scaled.shape)
    print(y_train_scaled.shape)


    # Use a large batch size to keep the GPU near 100% work load
    batch_size = 256

    # Sequence length is each random sequence for the length of time you set
    sequence_length = 24

    #With our sequence length and batch size set up we can now generate the sequence batch
    generator = batch_generator(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_train=num_train,
        x_train_scaled=x_train_scaled,
        y_train_scaled=y_train_scaled,
        num_x_signals=num_x_signals,
        num_y_signals=num_y_signals
    )
    # Tests the batch generator to see if it works
    x_batch, y_batch = next(generator)

    print(x_batch.shape)
    print(y_batch.shape)
    batch = 0  # First sequence in the batch
    signal = 0  # First signal from the 20 input-signals
    seq = x_batch[batch, :, signal]
    plt.plot(seq)
    plt.show()
    seq = y_batch[batch, :, signal]
    plt.plot(seq)
    plt.show()

    validation_data = (np.expand_dims(x_test_scaled, axis=0),
                       np.expand_dims(y_test_scaled, axis=0))

    # CREATING THE RECURRENT NEURAL NETWORK
    model = Sequential()
    # Add a Gated Recurrent Network to the network, has 512 outputs for each time-step
    # Input shape requires the shape of its input, None means of arbitrary length
    model.add(GRU(units=512,
                  return_sequences=True,
                  input_shape=(None, num_x_signals,)))
    # As output signals are limited between 0 and 1 we must do the same for the output from the neural network
    model.add(Dense(num_y_signals, activation='sigmoid'))

    if False:
        from tensorflow.python.keras.initializers import RandomUniform

        init = RandomUniform(minval=-0.05, maxval=0.05)

        model.add(Dense(num_y_signals,
                        activation='linear',
                        kernel_initializer=init))

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

    """
    model.fit_generator(generator=generator,
                        epochs=20,
                        steps_per_epoch=100,
                        validation_data=validation_data,
                        callbacks=callbacks)
    """

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

    if False:
        for res, metric in zip(result, model.metrics_names):
            print("{0}: {1:.3e}".format(metric, res))

    plot_comparison(start_idx=100000, length=1000, train=True,x_train_scaled=x_train_scaled, x_test_scaled=x_test_scaled, y_train=y_train,y_test=y_test,y_scaler=y_scaler,model=model)
