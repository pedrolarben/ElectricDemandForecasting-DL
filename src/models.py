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


def tcn(input_shape, output_size=1, optimizer='adam', loss='mae', nb_filters=64, kernel_size=2, nb_stacks=1,
        dilations=[1, 2, 4, 8, 16, 32], dropout_rate=0, use_skip_connections=True, use_batch_norm=False,
        activation='linear', return_sequences=False, dense_layers=[], dense_dropout=0.):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])

    x = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations,
            use_skip_connections=use_skip_connections, dropout_rate=dropout_rate, activation=activation,
            use_batch_norm=use_batch_norm, return_sequences=return_sequences)(inputs)
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)

    # Dense block
    x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)
    outputs = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)

    return model

