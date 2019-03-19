# -*- coding: utf-8 -*-
import numpy as np

# ----------    Simple DataFrame class replace the function of pandas.DataFrame---------- #
class DataFrame:
    def __init__(self, values, index, labels, father_pointer=1):
        """
        Initialize class DataFrame

        :param values: numpy.array -- values of data
        :param index: list of integer -- row index
        :param labels: list of string -- column labels
        :param father_pointer: DataFrame -- father Dataframe
        """
        self.values = values.copy() \
            if type(values) == type(np.array([])) else np.array(values)
        self.index  = index          # still use list to store number index
        self.labels = labels         # still use list to store string labels
        self.father_pointer = self \
            if father_pointer == 1 else father_pointer          # refer to the origin DataFrame
        self.basis = self.father_pointer.index[0]

    def __getitem__(self,mark):
        """
        []operation

        :param mark: list/integer/string -- row index or column labels
        :return: numpy.array -- values of data in corresponding places
        """
        # if mark is a list
        if type(mark) == type([]):
            if type(mark[0]) == type('string'):
                column = []
                for label in mark:
                    column = column + [i for i, place in enumerate(self.labels) if place == label]    # find the index of the label
                return_values = self.values[:, column].squeeze() \
                    if self.values.ndim != 1 else self.values[column].squeeze()          # if value is one dimension
                return return_values
            elif type(mark[0]) == type(0) or type(mark[0]) == type(0.0):                 # pick a row using number label
                row = []
                for label in mark:
                    row = row + [i for i, place in enumerate(self.index) if place == label]
                return_values = self.values[row, :].squeeze() \
                    if self.values.ndim != 1 else self.values[row].squeeze()
                return return_values
        # if mark is not a list
        elif type(mark) == type('string'):                      # pick a column using string label
                column = [i for i, place in enumerate(self.labels) if place == mark]      # find the index of the label
                column = np.array(column)[0]                    # for only 1 line, other wise don't need the [0]
                return_values = self.values[:, column].squeeze() \
                    if self.values.ndim != 1 else self.values[column].squeeze()
                return return_values
        elif type(mark) == type(0) or type(mark) == type(0.0):  # pick a row using number label
            row = [i for i, place in enumerate(self.index) if place == mark]
            row =np.array(row)[0]
            return_values = self.values[row, :].squeeze() \
                if self.values.ndim != 1 else self.values[row].squeeze()
            return return_values

    def __setitem__(self, key, value):
        """
        Assignment overloaded function: a[x] = b

        :param key: string -- column label
        :param value: real number -- value for assignment
        :return: DataFrame -- DataFrame after assignment
        """
        column = [i for i, place in enumerate(self.labels) if place == key]
        column = np.array(column)[0]    # not need [0], because it is an assignment
        self.values[:, column] = value  # change the value of itself
        temp = np.array(self.index)
        temp = temp - self.basis
        self.father_pointer.values[temp, column] = value        # change the values in father

    # ----------  public methods  ---------- #
    def copy(self):
        """
        Deep copy of class instance

        :return: DataFrame -- the copy of the DataFrame
        """
        values = self.values.copy()
        index = self.index
        labels = self.labels
        return DataFrame(values, index, labels)

    def reset_index(self):
        """Reset the index as [1,2,3...]"""
        self.index = range(self.values.shape[1])

    def drop(self, mark):
        """
        Drop columns or rows

        :param mark: list/integer/string -- row index or column labels
        :return: DataFrame -- DataFrame after dropping
        """
        values = self.values.copy()                 # not inplace: not change self
        # drop columns using string list or single string
        if type(mark[0]) == type('string'):
            mark = mark if type(mark) == type([]) else [mark]   # make it feasible for one column
            labels = np.array(self.labels)
            column = []
            for label in mark:
                column = column + [i for i, place in enumerate(self.labels) if place == label]
            values = np.delete(values, column, 1)               # delete values
            labels = np.delete(labels, column)
            labels = labels.tolist()
            sub_dataframe = DataFrame(values.squeeze(), self.index, labels)             # index is not changed
            return sub_dataframe
        # drop rows using index number
        elif type(mark[0]) == type(0) or type(mark[0]) == type(0.0) or \
                type(mark) == type(0) or type(mark) == type(0.0):
            mark = mark if type(mark[0]) == type(0) or type(mark[0]) == type(0.0) else [mark]  # make it feasible for one column
            index = np.array(self.index)
            row = []
            for label in mark:
                row = row + [i for i, place in enumerate(self.index) if place == label]
            values = np.delete(values, row, 0)                  # delete values
            index = np.delete(index, row)                       # delete index
            index = index.tolist()
            sub_dataframe = DataFrame(values.squeeze(), index, self.labels)
            return sub_dataframe

    def loc(self, position):
        """
        loc() operation to index/slice row data

        :param position: integer -- row index
        :return: DataFrame -- DataFrame after slicing
        """
        if type(position) == type(0) or type(position) == type(0.0) or type(position) == type([]):
            position = np.array(position)
            position = position - self.basis
            values = self.father_pointer.values[position]   # for number position
            temp = np.array(self.father_pointer.index)
        elif type(position) == np.ndarray and type(position[0]) == np.int64:
            position = position - self.basis
            values = self.father_pointer.values[position]   # for number position
            temp = np.array(self.father_pointer.index)
        elif type(position) == np.ndarray and type(position[0]) == np.bool_:
            values = self.values[position]
            temp = np.array(self.index)
        index = temp[position].tolist()                     # store the picked index relative to father
        sub_dataframe = DataFrame(values.squeeze(), index, self.labels, self.father_pointer)
        return sub_dataframe

    def value_counts(self, label):
        """
        count numbers of each different value in one column

        :param label: string -- column label
        :return: DataFrame -- DataFrame of counting results
        """
        count_num = []
        value = self.__getitem__(label)
        key = np.unique(value) # find unique value
        for k in key:
            mask = (value == k)
            y_new = value[mask]
            count_num.append(y_new.shape[0])
        key = key.tolist()
        order_dict = dict(zip(count_num, key))
        count_num = sorted(count_num,reverse=True)          # use dictionary to build the order of index
        for i in range(len(key)):
            key[i] = order_dict[count_num[i]]
        return DataFrame(np.array(count_num).squeeze(), key, [label])

# ----------------------       data processing            --------------------- #

def log_process(x):
    """
    Take log operation to data

    :param x: numpy.array -- data values
    :return: numpy.array -- data values after log process
    """
    columns = (sum(x < 0) == 0)                             # columns with no zero
    x_log = np.log(1/(1 + x[:, columns]))                     # to avoid log(0)
    x = np.hstack((x, x_log))
    return x

def build_poly(x, degrees):
    """
    Build polynomial features of data

    :param x: numpy.array -- data values
    :param degree: integer -- maximum degree
    :return: numpy.array -- data with poly values
    """
    tmp_dict = {}
    count = 0
    # build x[i] * x[j] polynomial features
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            tmp = x[:, i] * x[:, j]
            tmp_dict[count] = [tmp]
            count += 1
    poly_len = x.shape[1] * (degrees + 1) + count
    poly = np.zeros((x.shape[0], poly_len))

    for i in range(count):
        poly[:, x.shape[1] * degrees + i] = tmp_dict[i][0]
    # build x[i]^degree polynomial features
    for degree in range(1, degrees + 1):
        for i in range(x.shape[1]):
            poly[:, i + (degree - 1) * x.shape[1]] = np.power(x[:, i], degree)
    # build sqrt(x[i]) polynomial features
    for i in range(x.shape[1]):
        poly[:, i + x.shape[1] * degrees + count] = np.sqrt(np.abs(x[:, i]))
    # add 1 (const coefficiency )
    poly = np.hstack((np.ones((poly.shape[0], 1)), poly))

    return poly

def standardize(x):
    """
    Standardize the original data set

    :param x: numpy.array -- data values
    :return: numpy.array -- data values after standardization
    """
    x_standardize = np.empty_like(x, dtype='float64')

    mean_x = np.mean(x, axis=0)
    for colomn in range(x.shape[1]):
        x_standardize[:, colomn] = x[:, colomn] - mean_x[colomn]

    std_x = np.std(x, axis=0)
    for colomn in range(x.shape[1]):
        x_standardize[:, colomn] = x_standardize[:, colomn] / std_x[colomn]

    return x_standardize, mean_x, std_x

def compute_accuracy(y_pred,y_true):
    """
    calculate the accuracy

    :param y_pred: list -- list of predicted labals
    :param y_true: list -- list of true labals
    :return: float -- accuracy rate
    """
    return (1 - sum(abs(y_pred-y_true)/2)/len(y_pred))
    
# ----------     cross validation   ---------- #
def predict_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix

    :param weights: list of float -- model parameter w
    :param data: numpy.array -- data
    :return: list -- predicted label
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold

    :param y: numpy.array -- prediction labels
    :param k_fold: integer -- fold number
    :param seed: integer -- random seed
    :return: numpy.array -- shuffled index of each data fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, regression_method, **kwargs):
    """
    cross validation

    :param y: numpy.array -- prediction labels
    :param x: numpy.array -- data
    :param k_indices: numpy.array -- shuffled index of this data fold
    :param k: integer -- index of fold
    :param regression_method: function -- function name of regression
    :param kwargs: dict -- value of some parameter, e.g lambda
    :return: weight: list of float -- model parameter w
             accuracy_train: float -- accuracy rate of training data
             accuracy_test: float -- accuracy rate of test data
    """
    test_idx = k_indices[k]
    train_idx = list(set(np.arange(0,len(y)))-set(k_indices[k]))
    [x_train, y_train, x_test, y_test] = [x[train_idx], y[train_idx], x[test_idx], y[test_idx]]
    
    weight,loss = regression_method(y=y_train, tx=x_train, **kwargs)
    
    y_train_pred = predict_labels(weight, x_train)
    y_test_pred = predict_labels(weight, x_test)

    accuracy_train = compute_accuracy(y_train_pred, y_train)
    accuracy_test = compute_accuracy(y_test_pred, y_test)

    return weight, accuracy_train, accuracy_test

def cv_loop(y, x, k_fold, seed, regression_method, **kwargs):
    """
    cross validation loop using every fold of data as test data

    :param y: numpy.array -- prediction labels
    :param x: numpy.array -- data
    :param k_fold: integer -- number of the fold
    :param seed: integer -- random seed
    :param regression_method: function -- function name of regression
    :param kwargs: dict -- value of some parameter, e.g lambda
    :return: weight/k_fold: list of float -- average model parameter w
             accuracy_train: float -- average accuracy rate of training data
             accuracy_test: float -- average accuracy rate of test data
    """
    k_indices = build_k_indices(y, k_fold, seed)
    weight = np.zeros(x.shape[1])
    list_accuracy_train = []
    list_accuracy_test = []
    
    for k in range(k_fold):
        w, acc_tr, acc_te = cross_validation(y, x, k_indices, k, regression_method, **kwargs)
        weight = weight + w
        list_accuracy_train.append(acc_tr)
        list_accuracy_test.append(acc_te)
        # print("{} fold cv: Training accuracy: {} - Test accuracy : {}".format(k, acc_tr, acc_te))
    return weight/k_fold, np.mean(list_accuracy_train), np.mean(list_accuracy_test)

