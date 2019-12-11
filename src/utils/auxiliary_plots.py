import matplotlib.pyplot as plt


def plot_ts_forecasts(x_test, y_test, test_forecast, num_plots=4):
    for i in range(num_plots):
        x, y, forecast = x_test[i], y_test[i], test_forecast[i]
        plt.figure()
        plt.plot(x, label='History')
        plt.plot(list(range(len(x), len(x) + len(y))), y, 'go--', label='True future')
        plt.plot(list(range(len(x), len(x) + len(y))), forecast, 'rx--', label='Model prediction')
        plt.legend()
        plt.show()


def plot_training_history(history, keys):
    for k in keys:
        plt.figure()
        plt.plot(history.history[k])
        plt.plot(history.history['val_'+k])
        plt.title('model '+k)
        plt.ylabel(k)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()