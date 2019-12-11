import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.utils import auxiliary_plots, metrics
from src.preprocessing import normalization, data_generation

TRAIN_FILE_NAME = 'data/hourly_20140102_20191101_train.csv'
TEST_FILE_NAME = 'data/hourly_20140102_20191101_test.csv'

RESULT_FILE_NAME = 'files/results/experimental_results.csv'

SHOW_PLOTS = False

SEED = 1
tf.random.set_seed(SEED)
np.random.seed(SEED)

FORECAST_HORIZON = 24
PAST_HISTORY = [24, 48, 72, 96, 120, 144, 168]

BATCH_SIZE = [16, 36, 64, 128]
BUFFER_SIZE = 10000

EPOCHS = [25, 50, 100, 200]

results = []

# Read train file
with open(TRAIN_FILE_NAME, 'r') as datafile:
    ts_train = datafile.readlines()[1:]  # skip the header
    ts_train = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in ts_train])
    ts_train = np.reshape(ts_train, (ts_train.shape[0],))

# Read test data file
with open(TEST_FILE_NAME, 'r') as datafile:
    ts_test = datafile.readlines()[1:]  # skip the header
    ts_test = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in ts_test])
    ts_test = np.reshape(ts_test, (ts_test.shape[0],))

# Train/validation split
TRAIN_SPLIT = int(ts_train.shape[0]*0.8)

# Normalize training data
norm_params = normalization.get_normalization_params(ts_train[:TRAIN_SPLIT])
ts_train = normalization.normalize(ts_train, norm_params)
# Normalize test data with train params
ts_test = normalization.normalize(ts_test, norm_params)

for past_history, batch_size, epochs in tqdm(list(itertools.product(PAST_HISTORY, BATCH_SIZE, EPOCHS))):
    # Get x and y for training and validation
    x_train, y_train = data_generation.univariate_data(ts_train, 0, TRAIN_SPLIT, past_history, FORECAST_HORIZON)
    x_val, y_val = data_generation.univariate_data(ts_train, TRAIN_SPLIT-past_history, ts_train.shape[0], past_history, FORECAST_HORIZON)
    # Get x and y for test data
    x_test, y_test = data_generation.univariate_data(ts_test, 0, ts_test.shape[0], past_history, FORECAST_HORIZON)

    # Convert numpy data to tensorflow dataset
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache().shuffle(BUFFER_SIZE).batch(batch_size).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).repeat()
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


    # Create model
    model_list = {
        'lstm': simple_lstm(x_train.shape, FORECAST_HORIZON, 'adam', 'mae'),
        'mlp': mlp(x_train.shape, FORECAST_HORIZON, 'adam', 'mae', hidden_layers=[64,32,16]),
        'cldnn': cldnn(x_train.shape, FORECAST_HORIZON, 'adam', 'mae'),
        'cnn': cnn(x_train.shape, FORECAST_HORIZON, 'adam', 'mae'),
        # 'tcn': tcn(x_train.shape, FORECAST_HORIZON, 'adam', 'mae'),
    }

    evaluation_interval = int(np.ceil(x_train.shape[0] / batch_size))

    for model_name, model in model_list.items():
        print(model.summary())

        # Train the model
        # TODO: Callbacks
        history = model.fit(train_data, epochs=epochs,
                              steps_per_epoch=evaluation_interval,
                              validation_data=val_data, validation_steps=evaluation_interval)

        # Plot training and evaluation loss evolution
        if SHOW_PLOTS:
            auxiliary_plots.plot_training_history(history, ['loss'])

        # Get validation results
        val_forecast = model.predict(x_val)
        val_forecast = normalization.denormalize(val_forecast, norm_params)
        y_val = normalization.denormalize(y_val, norm_params)

        val_metrics = metrics.evaluate_all(y_val, val_forecast)
        print('Validation metrics', val_metrics)

        # TEST
        # Predict with test data and get results
        test_forecast = model.predict(test_data)

        test_forecast = normalization.denormalize(test_forecast, norm_params)
        y_test = normalization.denormalize(y_test, norm_params)
        x_test = normalization.denormalize(x_test, norm_params)

        test_metrics = metrics.evaluate_all(y_test, test_forecast)
        print('Test scores', test_metrics)

        # Plot some test predictions
        if SHOW_PLOTS:
            auxiliary_plots.plot_ts_forecasts(x_test, y_test, test_forecast)

        # Save results
        val_metrics = {'val_'+k: val_metrics[k] for k in val_metrics}
        model_metric = {'MODEL': model_name,
                        'FORECAST_HORIZON': FORECAST_HORIZON,
                        'PAST_HISTORY': past_history,
                        'BATCH_SIZE': batch_size,
                        'EPOCHS': epochs,
                        **test_metrics,
                        **val_metrics,
                        **history.history
                        }
        results.append(model_metric)


results = pd.DataFrame(results).to_csv(RESULT_FILE_NAME, index=False)


