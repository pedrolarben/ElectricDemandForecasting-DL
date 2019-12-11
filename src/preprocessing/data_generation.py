import numpy as np
from preprocessing.normalization import get_normalization_params, normalize


def read_ts_dataset(filename):
    """
    Function for reading csv dataset with multiple time series and storing them into an array
    :param filename:
    :return: array of time series
    """
    with open(filename, 'r') as datafile:
        data = datafile.readlines()
        data = np.asarray([np.asarray(l.rstrip().split(','), dtype=np.float32) for l in data])
    return data


def univariate_data(tseries, start_index, end_index, history_size, forecast_horizon, step=1, single_step=False):
    """
    Function to extract input-output windows from a time series
    :param tseries: time series array
    :param start_index: first value of the first x (input window)
    :param end_index: first value of the last y (output window)
    :param history_size: size of input window
    :param forecast_horizon: size of output prediction window
    :param step: sampling of time series for creating input windows
    :param single_step: single or multi-step prediction window
    :return: data (x), labels (y)
    """

    data = []
    labels = []

    start_index = start_index + history_size

    for i in range(start_index, end_index - forecast_horizon + 1):
        indices = range(i - history_size, i, step)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(tseries[indices], (history_size, 1)))

        if single_step:
            labels.append(tseries[i + forecast_horizon])
        else:
            labels.append(tseries[i:i + forecast_horizon])

    return np.array(data), np.array(labels)


def get_train_test_data(ts_list, test_ts_list, past_history, forecast_horizon, val_split_func=None,
                        norm_method='zscore'):
    """
    Obtain train, validation and test datasets
    :param ts_list: array with train part of time series dataset
    :param test_ts_list: array with test part of time series dataset
    :param past_history: size of input window (x)
    :param forecast_horizon: size of output window (y)
    :param val_split_func: function to calculate length of train/val split with respect to length of time series
    :param norm_method: zscore or minmax
    :return: x_train, y_train, x_val, y_val, x_test, y_test, and normalization parameters for each time series
    """

    x_train, y_train, x_val, y_val, x_test, y_test = ([] for _ in range(6))
    norm_params_list = []

    for i in range(len(ts_list)):

        # Get time series data by index
        ts = ts_list[i]
        test_ts = np.concatenate((ts[-past_history:], test_ts_list[i]))

        # Length of time series
        len_ts = ts.shape[0]
        len_test_ts = test_ts.shape[0]

        # Train/validation split (default -forecast_horizon)
        train_split = val_split_func(len_ts) if val_split_func else len_ts

        # Normalize data
        norm_params = get_normalization_params(ts[:train_split])
        norm_params_list.append(norm_params)
        ts = normalize(ts, norm_params, norm_method)
        test_ts = normalize(test_ts, norm_params, norm_method)

        # Training data
        ts_x_train, ts_y_train = univariate_data(ts, 0, train_split, past_history, forecast_horizon)
        x_train.extend(ts_x_train)
        y_train.extend(ts_y_train)

        # Validation data
        if train_split < len_ts:
            ts_x_val, ts_y_val = univariate_data(ts, train_split - past_history, len_ts, past_history, forecast_horizon)
            x_val.extend(ts_x_val)
            y_val.extend(ts_y_val)

        # Test data
        ts_x_test, ts_y_test = univariate_data(test_ts, 0, len_test_ts, past_history, forecast_horizon)
        x_test.extend(ts_x_test)
        y_test.extend(ts_y_test)

    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), \
        np.array(x_test), np.array(y_test), norm_params_list
