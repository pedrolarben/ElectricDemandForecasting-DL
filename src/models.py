import tensorflow as tf
from tcn import TCN


def lstm(input_shape, output_size=1, optimizer='adam', loss='mae', num_stack_layers=1, units=50, dropout=0, hidden_dense_layers=[]):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    return_sequences = num_stack_layers>1
    x = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(inputs)
    for i in range(num_stack_layers-1):
        return_sequences = i<num_stack_layers-2
        x = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(x)
    for hidden_units in hidden_dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def gru(input_shape, output_size=1, optimizer='adam', loss='mae', hidden_recurrent_layers=[50], hidden_dense_layers=[],
        return_sequences=True):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.LSTM(hidden_recurrent_layers.pop(0))(inputs)
    for hidden_units in hidden_recurrent_layers:
        x = tf.keras.layers.GRU(hidden_units, return_sequences=return_sequences)(x)
    for hidden_units in hidden_dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def ernn(input_shape, output_size=1, optimizer='adam', loss='mae', hidden_recurrent_layers=[50], hidden_dense_layers=[],
         return_sequences=True):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.LSTM(hidden_recurrent_layers.pop(0))(inputs)
    for hidden_units in hidden_recurrent_layers:
        x = tf.keras.layers.SimpleRNN(hidden_units, return_sequences=return_sequences)(x)
    for hidden_units in hidden_dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def mlp(input_shape, output_size=1, optimizer='adam', loss='mae', hidden_layers=[32, 16, 8]):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.Flatten()(inputs)  # Convert the 2d input in a 1d array
    for hidden_units in hidden_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
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


def tcn(input_shape, output_size=1, optimizer='adam', loss='mae', nb_filters=64, kernel_size=2, nb_stacks=1,
        dilations=[1, 2, 4, 8, 16, 32], use_skip_connections=True, dropout_rate=0, use_batch_norm=False, activation='linear',):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])

    x = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations,
            use_skip_connections=use_skip_connections, dropout_rate=dropout_rate, activation=activation,
            use_batch_norm=use_batch_norm)(inputs)
    outputs = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
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
