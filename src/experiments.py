import itertools
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models import lstm, cnn, mlp, cldnn, tcn
from utils import auxiliary_plots, metrics
from utils.print_functions import notify_slack
from preprocessing import normalization, data_generation
import time

METRICS = ['mse', 'rmse', 'nrmse', 'mae', 'mpe', 'mape', 'mdape', 'smape', 'smdape',
           'mase', 'rmspe', 'rmsse', 'mre', 'rae', 'mrae', 'std_ae', 'std_ape']

TCN_PARAMS = {
    'nb_filters': [32, 64, 128],
    'kernel_size': [2, 3, 4, 5],
    'nb_stacks': [1, 2, 3, 4],
    'dilations': [[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32, 64],
                  [1, 3, 6], [1, 3, 6, 12], [1, 5, 7], [1, 5, 7, 14]],
    'dropout_rate': [0],
}
LSTM_PARAMS = {
    'num_stack_layers': [1, 2, 3],
    'units': [32, 64, 128],
    'dropout': [0]
}


def run_experiments(train_file_name, test_file_name, result_file_name, forecast_horizon, past_history_ls, batch_size_ls,
                    epochs_ls, tcn_params=TCN_PARAMS, lstm_params=LSTM_PARAMS, gpu_number=None, metrics_ls=METRICS,
                    buffer_size=1000, seed=1, show_plots=False, webhook=None, validation_size=0.2):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    device_name = str(gpus)
    if len(gpus) >= 2 and gpu_number is not None:
        device = gpus[gpu_number]
        tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_visible_devices(device, 'GPU')
        device_name = str(device)
        print(device)

    # Write result csv header
    current_index = 0
    try:
        with open(result_file_name, 'r') as resfile:
            current_index = sum(1 for line in resfile) - 1
    except IOError:
        pass
    print('CURRENT INDEX', current_index)
    if current_index == 0:
        with open(result_file_name, 'w') as resfile:
            resfile.write(';'.join([str(a) for a in
                                    ['MODEL', 'MODEL_DESCRIPTION', 'FORECAST_HORIZON', 'PAST_HISTORY', 'BATCH_SIZE',
                                     'EPOCHS'] +
                                    metrics_ls + ['val_' + m for m in metrics_ls] + ['loss', 'val_loss',
                                                                                     'Execution_time',
                                                                                     'Device']]) + "\n")

    # Read train file
    with open(train_file_name, 'r') as datafile:
        ts_train = datafile.readlines()[1:]  # skip the header
        ts_train = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in ts_train])
        ts_train = np.reshape(ts_train, (ts_train.shape[0],))

    # Read test data file
    with open(test_file_name, 'r') as datafile:
        ts_test = datafile.readlines()[1:]  # skip the header
        ts_test = np.asarray([np.asarray(l.rstrip().split(',')[0], dtype=np.float32) for l in ts_test])
        ts_test = np.reshape(ts_test, (ts_test.shape[0],))

    # Train/validation split
    TRAIN_SPLIT = int(ts_train.shape[0] * (1 - validation_size))
    print(ts_train.shape, TRAIN_SPLIT)
    # Normalize training data
    norm_params = normalization.get_normalization_params(ts_train[:TRAIN_SPLIT])
    ts_train = normalization.normalize(ts_train, norm_params)
    # Normalize test data with train params
    ts_test = normalization.normalize(ts_test, norm_params)

    i = 0
    index_1, total_1 = 0, len(list(itertools.product(past_history_ls, batch_size_ls, epochs_ls)))
    for past_history, batch_size, epochs in tqdm(list(itertools.product(past_history_ls, batch_size_ls, epochs_ls))):
        index_1 += 1
        # Get x and y for training and validation
        x_train, y_train = data_generation.univariate_data(ts_train, 0, TRAIN_SPLIT, past_history, forecast_horizon)
        x_val, y_val = data_generation.univariate_data(ts_train, TRAIN_SPLIT - past_history, ts_train.shape[0],
                                                       past_history, forecast_horizon)
        print(x_train.shape, y_train.shape, '\n', x_val.shape, y_val.shape)
        # Get x and y for test data
        x_test, y_test = data_generation.univariate_data(ts_test, 0, ts_test.shape[0], past_history, forecast_horizon)

        # Convert numpy data to tensorflow dataset
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache().shuffle(buffer_size).batch(
            batch_size).repeat()
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(
            batch_size).repeat() if validation_size > 0 else None
        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        # Create models
        model_list = {}
        model_description_list = {}
        if tcn_params is not None:
            model_list = {'TCN_{}'.format(j): (tcn, [x_train.shape, forecast_horizon, 'adam', 'mae', *params]) for
                          j, params in
                          enumerate(itertools.product(*tcn_params.values())) if
                          params[1] * params[2] * params[3][-1] == past_history}
            model_description_list = {'TCN_{}'.format(j): str(dict(zip(tcn_params.keys(), params))) for j, params in
                                      enumerate(itertools.product(*tcn_params.values())) if
                                      params[1] * params[2] * params[3][-1] == past_history}
        if lstm_params is not None:
            model_list = {**model_list,
                          **{'LSTM_{}'.format(j): (lstm, [x_train.shape, forecast_horizon, 'adam', 'mae', *params]) for
                             j, params in
                             enumerate(itertools.product(*lstm_params.values()))}}
            model_description_list = {**model_description_list,
                                      **{'LSTM_{}'.format(j): str(dict(zip(lstm_params.keys(), params))) for j, params
                                         in
                                         enumerate(itertools.product(*lstm_params.values()))}}

        steps_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))
        validation_steps = steps_per_epoch if val_data else None

        index_2, total_2 = 0, len(model_list.keys())
        for model_name, (model_function, params) in tqdm(model_list.items(), position=1):
            index_2 += 1
            i += 1
            if i <= current_index:
                continue
            start = time.time()
            model = model_function(*params)
            print(model.summary())

            # Train the model
            history = model.fit(train_data, epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_data, validation_steps=validation_steps)

            # Plot training and evaluation loss evolution
            if show_plots:
                auxiliary_plots.plot_training_history(history, ['loss'])

            # Get validation results
            val_metrics = {}
            if validation_size > 0:
                val_forecast = model.predict(x_val)
                val_forecast = normalization.denormalize(val_forecast, norm_params)
                y_val_denormalized = normalization.denormalize(y_val, norm_params)

                val_metrics = metrics.evaluate(y_val_denormalized, val_forecast, metrics_ls)
                print('Validation metrics', val_metrics)

            # TEST
            # Predict with test data and get results
            test_forecast = model.predict(test_data)

            test_forecast = normalization.denormalize(test_forecast, norm_params)
            y_test_denormalized = normalization.denormalize(y_test, norm_params)
            x_test_denormalized = normalization.denormalize(x_test, norm_params)

            test_metrics = metrics.evaluate(y_test_denormalized, test_forecast, metrics_ls)
            print('Test scores', test_metrics)

            # Plot some test predictions
            if show_plots:
                auxiliary_plots.plot_ts_forecasts(x_test_denormalized, y_test_denormalized, test_forecast)

            # Save results
            val_metrics = {'val_' + k: val_metrics[k] for k in val_metrics}
            model_metric = {'MODEL': model_name,
                            'MODEL_DESCRIPTION': model_description_list[model_name],
                            'FORECAST_HORIZON': forecast_horizon,
                            'PAST_HISTORY': past_history,
                            'BATCH_SIZE': batch_size,
                            'EPOCHS': epochs,
                            **test_metrics,
                            **val_metrics,
                            **history.history,
                            'Execution_time': time.time() - start,
                            'Device': device_name
                            }

            notify_slack('Progress: {0}/{1} ({2}/{3}) \nMetrics:{4}'.format(index_1, total_1, index_2, total_2,
                                                                            str({'Model': model_name,
                                                                                 'WAPE': str(test_metrics['wape']),
                                                                                 'Execution_time': "{0:.2f}  seconds".format(
                                                                                     time.time() - start)
                                                                                 })), webhook=webhook)

            with open(result_file_name, 'a') as resfile:
                resfile.write(';'.join([str(a) for a in model_metric.values()]) + "\n")
