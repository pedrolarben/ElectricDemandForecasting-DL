import tensorflow as tf


def simple_lstm(input_shape, output_size=1, optimizer='adam', loss='mae'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape=input_shape[-2:]),
        tf.keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=optimizer, loss=loss)

    return model
