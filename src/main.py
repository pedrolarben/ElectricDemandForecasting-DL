from experiments import run_experiments
import os

WEBHOOK = os.environ.get('webhook_slack')
print(WEBHOOK)

METRICS = ['mse', 'rmse', 'nrmse', 'mae', 'wape', 'mpe', 'mape', 'mdape', 'smape', 'smdape',
           'mase', 'rmspe', 'rmsse', 'mre', 'rae', 'mrae', 'std_ae', 'std_ape']

TCN_PARAMS = {
    'nb_filters': [32, 64, 128],
    'kernel_size': [2, 3, 4, 5, 6],
    'nb_stacks': [1, 2, 3, 4, 5],
    'dilations': [[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32, 64],
                  [1, 3, 9], [1, 3, 9, 27], [1, 3, 9, 27, 81],
                  [1, 3, 6], [1, 3, 6, 12], [1, 3, 6, 12, 24], [1, 3, 6, 12, 24], [1, 3, 6, 12, 24, 48],
                  [1, 5, 7], [1, 5, 7, 14], [1, 5, 7, 14, 28], [1, 5, 7, 14, 28, 56]],
    'dropout_rate': [0],
}
LSTM_PARAMS = {
    'num_stack_layers': [1, 2, 3],
    'units': [32, 64, 128],
    'dropout': [0]
}

FORECAST_HORIZON = 24
PAST_HISTORY = [144, 168, 288]

BATCH_SIZE = [64, 128, 256]
BUFFER_SIZE = 10000

EPOCHS = [25, 50, 100]

_GPU_NUMBER = None

# Electric demand forecasting
run_experiments(train_file_name='data/hourly_20140102_20191101_train.csv',
                test_file_name='data/hourly_20140102_20191101_test.csv',
                result_file_name='files/results/experimental_results_electricity.csv',
                forecast_horizon=FORECAST_HORIZON,
                past_history_ls=PAST_HISTORY,
                batch_size_ls=BATCH_SIZE,
                epochs_ls=EPOCHS,
                tcn_params=TCN_PARAMS,
                lstm_params=LSTM_PARAMS,
                gpu_number=_GPU_NUMBER,
                metrics_ls=METRICS,
                buffer_size=1000,
                seed=1,
                show_plots=False,
                webhook=WEBHOOK,
                validation_size=0.)

# Electric vehicle power consumption forecasting
run_experiments(train_file_name='data/CECOVEL_train.csv',
                test_file_name='data/CECOVEL_test.csv',
                result_file_name='files/results/experimental_results_EV.csv',
                forecast_horizon=FORECAST_HORIZON,
                past_history_ls=PAST_HISTORY,
                batch_size_ls=BATCH_SIZE,
                epochs_ls=EPOCHS,
                tcn_params=TCN_PARAMS,
                lstm_params=LSTM_PARAMS,
                gpu_number=None,
                metrics_ls=METRICS,
                buffer_size=1000,
                seed=1,
                show_plots=False,
                webhook=WEBHOOK,
                validation_size=0.)
