def print_dataset_shapes(x_train, y_train, x_val, y_val, x_test, y_test):
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

def print_sample_window(x_train,y_train):
    print('Sample single window of past history')
    print(x_train[0])
    print('Sample target to predict')
    print(y_train[0])