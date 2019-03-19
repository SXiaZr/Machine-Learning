# -*- coding: utf-8 -*-
import numpy as np
            
# ----------    Cost calculation    ---------- #
def calculate_mse(e):
    """
    Calculate the mse for vector e

    :param e: numpy.array -- error values
    :return: float -- values of lost function
    """
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """
    Calculate the mae for vector e

    :param e: numpy.array -- error values
    :return: float -- values of lost function
    """
    return np.mean(np.abs(e))

def compute_loss(y, tx, w, method = calculate_mse):
    """
    Calculate the loss according to loss function

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param w: numpy.array -- weights parameter of model
    :param method: function -- loss function
    :return: float -- values of lost function
    """
    e = y - tx.dot(w)
    return method(e)

# ----------    Gradient descent    ---------- #
def compute_gradient(y, tx, w):
    """
    Compute the gradient

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param w: numpy.array -- weights parameter of model
    :return: numpy.array -- gradient values
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param initial_w: numpy.array -- initial weights parameter of model
    :param max_iters: integer -- maximum iteration number
    :param gamma: float -- step size
    :return: w:  last weight parameter of model
             loss: last loss values
    """
    w = initial_w
    loss = 0.0
    for n_iter in range(max_iters): # range返回迭代器
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
    return w, loss

# ----------    Stochastic descent    ---------- #
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param batch_size: integer -- size of batch
    :param num_batches: integer -- number of batch
    :param shuffle: bool -- whether to shuffle the index
    :return:iterator -- gives mini-batches of `batch_size` matching elements from `y` and `tx`
    """
    data_size = y.shape[0]  # len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]  # Gets the view corresponding to the indices.
        shuffled_tx = tx[shuffle_indices]  # NB : iterable of arrays as indexing
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)  # avoid crossing the index border
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param initial_w: numpy.array -- initial weights parameter of model
    :param max_iters: integer -- maximum iteration number
    :param gamma: float -- step size
    :return: w:  last weight parameter of model
             loss: last loss values
    """
    w = initial_w
    loss = 0.0
    for i in range(max_iters):
        for y_batch, x_batch in batch_iter(y, tx, batch_size, 1, True): # Since num_batches=1, the loop run only once
            w = w - gamma * compute_gradient(y_batch, x_batch, w)
            loss = compute_loss(y, tx, w)                               # Every new w match a loss, ingore the original loss
    return w, loss

# ----------    Least squares    ---------- #
def least_squares(y, tx):
    """
    Calculate the least squares solution.

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :return: w:  last weight parameter of model
             loss: last loss values
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_loss(y, tx, w))
    return w, loss

# ----------    Ridge Regression    ---------- #
def ridge_regression(y, tx, lambda_):
    """
    implement ridge regression

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param lambda_: float -- penalty coefficient
    :return: w:  last weight parameter of model
             loss: last loss values
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss
# ----------    Logistic regression    ---------- #
def logistic(t):
    """
    apply logistic function on t

    :param t: numpy.array -- values
    :return: numpy.array -- logistic value
    """
    logistic_value = np.exp(t) / (1 + np.exp(t))
    logistic_value[np.isnan(logistic_value)] =1

    return logistic_value

def calculate_logi_loss(y, tx, w):
    """
    calculate loss of logistic function

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param w: numpy.array -- initial weights parameter of model
    :return: float -- loss values
    """
    pred = logistic(tx.dot(w))
    loss = y.T.dot(np.log(pred + 1e-12)) + (1 - y).T.dot(np.log(1 - pred + 1e-12))  # avoid 0
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """
    compute the gradient of loss.

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param w: numpy.array -- weights parameter of model
    :return: numpy.array -- gradient values
    """
    pred = logistic(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_logi_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return w,loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    implement logistic regression

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param initial_w: numpy.array -- initial weights parameter of model
    :param max_iters: integer -- maximum iteration number
    :param gamma: float -- step size
    :return: w:  last weight parameter of model
             loss: last loss values
    """
    # init parameters
    y = (y + 1) / 2  # make -1,1 values to 0,1
    threshold = 1e-5
    w = initial_w
    y = y.reshape(len(y), 1)
    losses = []
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w,loss = learning_by_gradient_descent(y, tx, w, gamma)
            # if iter % 10 == 0:
            #     print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w,losses[-1]

# ----------    Regularized logistic regression    ---------- #
def penalized_logistic_regression(y, tx, w, lambda_):
    loss = calculate_logi_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = tx.T.dot(1.0 / (1 + np.exp(-tx.dot(w))) - y) + 2 * lambda_ * w
    return loss, gradient

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """
    implement regularized logistic regression

    :param y: numpy.array -- y values of data
    :param tx: numpy.array -- x values of data
    :param lambda_: float -- penalty coefficient
    :param initial_w: numpy.array -- initial weights parameter of model
    :param max_iters: integer -- maximum iteration number
    :param gamma: float -- step size
    :return: w:  last weight parameter of model
             loss: last loss values
    """
    # Set default weight
    y = (y + 1) / 2  # make -1,1 values to 0,1
    threshold = 1e-5
    weight = initial_w
    losses = []
    for i in range(max_iters):
        # get loss and update w.
        loss, gradient = penalized_logistic_regression(y, tx, weight, lambda_)
        weight = weight - gamma * gradient
        losses.append(loss)
        # termination condition
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return weight,losses[-1]
