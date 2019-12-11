import tensorflow as tf
from tcn import TCN
from keras import  layers, Model


def simple_lstm(input_shape, output_size=1, optimizer='adam', loss='mae'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape=input_shape[-2:]),
        tf.keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=optimizer, loss=loss)

    return model


def mlp(input_shape, output_size=1, optimizer='adam', loss='mae', hidden_layers=[32,16,8]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape[-2:]))
    model.add(tf.keras.layers.Flatten())  # Convert the 2d input in a 1d array
    for hidden_units in hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_units))
    model.add(tf.keras.layers.Dense(output_size))
    model.compile(optimizer=optimizer, loss=loss)

    return model


def cnn(input_shape, output_size=1, optimizer='adam', loss='mae'):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.Conv1D(256, 9, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024)(x)
    outputs = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def tcn(input_shape, output_size=1, optimizer='adam', loss='mae'):
    # TODO fix: 'RepeatDataset' object has no attribute 'ndim'
    inputs = layers.Input(shape=input_shape[-2:])
    x = TCN()(inputs)
    #x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    outputs = layers.Dense(output_size)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)

    return model

def cldnn(input_shape, output_size=1, optimizer='adam', loss='mae'):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.Conv1D(256, 9, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(256, 4, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.Dense(1, activation='linear')(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, inputs])

    x = tf.keras.layers.LSTM(256, return_sequences=True, activation='tanh', recurrent_activation='sigmoid',
                             recurrent_dropout=0, unroll=False, use_bias=True)(x)
    x = tf.keras.layers.LSTM(256, return_sequences=True, activation='tanh', recurrent_activation='sigmoid',
                             recurrent_dropout=0, unroll=False, use_bias=True)(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, inputs])

    x = tf.keras.layers.Dense(1024, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(1024, activation='sigmoid')(x)

    x = tf.keras.layers.Flatten()(x)

    outputs = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)

    return model