import numpy as np


def normalize(data, norm_params, method='zscore'):
    """
    Normalize time series
    :param data: time series
    :param norm_params: tuple with params mean, std, max, min
    :param method: zscore or minmax
    :return: normalized time series
    """
    assert method in ['zscore', 'minmax', None]

    if method == 'zscore':
        return (data - norm_params['mean']) / norm_params['std']

    elif method == 'minmax':
        return (data - norm_params['min']) / (norm_params['max'] - norm_params['max'])

    elif method is None:
        return data

def denormalize(data, norm_params, method='zscore'):
    """
    Reverse normalization time series
    :param data: normalized time series
    :param norm_params: tuple with params mean, std, max, min
    :param method: zscore or minmax
    :return: time series in original scale
    """
    assert method in ['zscore', 'minmax', None]

    if method == 'zscore':
        return (data * norm_params['std']) + norm_params['mean']

    elif method == 'minmax':
        return (data * (norm_params['max']) - (norm_params['min']) + norm_params['max'])

    elif method is None:
        return data


def denormalize_dataset(dataset, norm_params_list, method='zscore'):
    """
    Reverse normalization on complete time series dataset
    :param dataset: array of all time series
    :param norm_params_list: array with tuples of normalization params for each time series
    :param method: zscore or minmax
    :return: array with all time series at original scale
    """
    return np.asarray([denormalize(dataset[i], norm_params_list[i], method) for i in range(dataset.shape[0])])


def get_normalization_params(data):
    """
    Obtain parameters for normalization
    :param data: time series
    :return: dict with string keys
    """
    norm_params = {}
    norm_params['mean'] = data.mean()
    norm_params['std'] = data.std()
    norm_params['max'] = data.max()
    norm_params['min'] = data.min()

    return norm_params
