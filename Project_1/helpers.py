# -*- coding: utf-8 -*-
import csv
import numpy as np

def load_csv_data(data_path, sub_sample=False):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)

    :param data_path: string -- data path
    :param sub_sample: bool -- whether to sub-sample
    :return: yb: numpy.array -- prediction labels
             input_data: numpy.array -- data
             ids: numpy.array -- id of data
             labels: list of string -- name of the features
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    labels = open(data_path,'r').readline()


    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]
    labels = labels.strip().split(",")
    del labels[0]
    del labels[0]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids, labels

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
