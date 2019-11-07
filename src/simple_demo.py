import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import simple_lstm
from src import metrics
from collections import defaultdict

tf.random.set_seed(1)
np.random.seed(1)

TRAIN_FILE_NAME = 'data/hourly_20140102_20191101_train.csv'
TEST_FILE_NAME = 'data/hourly_20140102_20191101_test.csv'

FORECAST_HORIZON = 12
PAST_HISTORY = 20

BATCH_SIZE = 32
BUFFER_SIZE = 10000


x_train, y_train, x_val, y_val, x_test, y_test = None, None, None, None, None, None
norm_params = []


# Normalization functions
def normalize(a, train_params):
    return (a - train_params[0]) / train_params[1]


def denormalize(a_ls, norm_params):
    return np.asarray([a_ls[i] * norm_params[i][1] + norm_params[i][0] for i in range(a_ls.shape[0])])


# Read test predictions
with open(TEST_FILE_NAME, 'r') as datafile:
    y_test = datafile.readlines()[1:]  # skip the header
    y_test = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in y_test])
    y_test = np.reshape(y_test, (1, y_test.shape[0]))

# Read train file
with open(TRAIN_FILE_NAME, 'r') as datafile:
    ts_list = datafile.readlines()[1:]  # skip the header
    ts_list = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in ts_list])
    ts_list = np.reshape(ts_list, (1, ts_list.shape[0]))

for i, ts in enumerate(ts_list):
    # Train test split
    TRAIN_SPLIT = ts.shape[0]-FORECAST_HORIZON

    # Normalize data
    train_mean = ts[:TRAIN_SPLIT].mean()
    train_std = ts[:TRAIN_SPLIT].std()
    train_max = ts[:TRAIN_SPLIT].max()

    norm_params.append((train_mean, train_std, train_max))

    ts = normalize(ts, (train_mean, train_std, train_max))

    # Normalize test data
    y_test[i] = normalize(y_test[i], (train_mean, train_std, train_max))

    ts_x_train = []
    ts_y_train = []

    for j in range(PAST_HISTORY, TRAIN_SPLIT):
        indices = range(j - PAST_HISTORY, j)
        # Reshape from (PAST_HISTORY,) to (PAST_HISTORY,1)
        ts_x_train.append(np.reshape(ts[indices], (PAST_HISTORY,1)))
        ts_y_train.append(ts[j: j + FORECAST_HORIZON])

    ts_x_val = []
    ts_y_val = []

    for j in range(TRAIN_SPLIT, len(ts) - (FORECAST_HORIZON - 1)):
        indices = range(j - PAST_HISTORY, j)
        # Reshape from (PAST_HISTORY,) to (PAST_HISTORY,1)
        ts_x_val.append(np.reshape(ts[indices], (PAST_HISTORY, 1)))
        ts_y_val.append(ts[j : j+FORECAST_HORIZON])

    ts_x_test = []
    indices = range(ts.shape[0] - PAST_HISTORY, ts.shape[0])
    ts_x_test.append(np.reshape(ts[indices], (PAST_HISTORY,1)))


    if x_train is None:
        x_train, y_train, x_val, y_val = np.array(ts_x_train), np.array(ts_y_train), np.array(ts_x_val), np.array(ts_y_val)
        x_test = np.array(ts_x_test)
    else:
        x_train = np.concatenate((x_train, ts_x_train))
        y_train = np.concatenate((y_train, ts_y_train))
        x_val = np.concatenate((x_val, ts_x_val))
        y_val = np.concatenate((y_val, ts_y_val))
        x_test = np.concatenate((x_test, ts_x_test))


print("TRAINING DATA")
print("Input shape", x_train.shape)
print("Output_shape", y_train.shape)
print()
print("VALIDATION DATA")
print("Input shape", x_val.shape)
print("Output_shape", y_val.shape)
print()
print("TEST DATA")
print("Input shape", x_test.shape)
print("Output_shape", y_test.shape)
print()

print('Sample single window of past history')
print(x_train[0])
print('Sample target to predict')
print(y_train[0])

# Create model

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.batch(BATCH_SIZE).repeat()

model = simple_lstm(x_train.shape, FORECAST_HORIZON, 'adam', 'mae')

# Let's make a sample prediction, to check the output of the model
for x, y in val_data.take(1):
    print("\nSample output", model.predict(x).shape)

# Train the model
EVALUATION_INTERVAL = np.ceil(x_train.shape[0]/BATCH_SIZE)
EPOCHS = 1

model.fit(train_data, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_data, validation_steps=EVALUATION_INTERVAL)

val_forecast = model.predict(x_val)
test_forecast = model.predict(x_test)

# Denormalize output
val_forecast = denormalize(val_forecast, norm_params)
y_val = denormalize(y_val, norm_params)

test_forecast = denormalize(test_forecast, norm_params)
y_test = denormalize(y_test, norm_params)
x_test = denormalize(x_test, norm_params)


val_scores = defaultdict(list)
for i in range(len(y_val)):
    val_scores['smape'].append(metrics.smape(y_val[i], val_forecast[i]))
    val_scores['mase'].append(metrics.mase(y_val[i], val_forecast[i], seasonality=1))
for k in val_scores.keys():
    val_scores[k] = [np.mean(val_scores[k])]
print('Validation scores', dict(val_scores))

test_scores = defaultdict(list)
for i in range(len(y_test)):
    test_scores['smape'].append(metrics.smape(y_test[i], test_forecast[i]))
    test_scores['mase'].append(metrics.mase(y_test[i], test_forecast[i], seasonality=1))
for k in test_scores.keys():
    test_scores[k] = [np.mean(test_scores[k])]
print('Test scores', dict(test_scores))

for i in range(4):
    x, y, forecast = x_test, y_test, test_forecast
    plt.figure()
    plt.plot(x[i], label='History')
    plt.plot(list(range(len(x[i]), len(x[i])+FORECAST_HORIZON)), y[i], 'go--', label='True future')
    plt.plot(list(range(len(x[i]), len(x[i])+FORECAST_HORIZON)), forecast[i], 'rx--', label='Model prediction')
    plt.legend()
    plt.show()


