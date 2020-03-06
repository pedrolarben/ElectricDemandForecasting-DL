import os
import requests
import json


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


def notify_slack(msg, webhook=None):
    if webhook is None:
        webhook = os.environ.get('webhook_slack')
    if webhook is not None:
        try:
            requests.post(webhook, json.dumps({'text': msg}))
        except:
            print('Error while notifying slack')
            print(msg)
    else:
        print("NO WEBHOOK FOUND")