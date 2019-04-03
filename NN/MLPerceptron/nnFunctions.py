import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle


def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label


def sig(z):
    return 1/(1 + np.exp(-z))

def sigmoid(z):
    # using numpy ufunc to vectorize function

    s = sig(z)

    return s

# weighed sums
def applyw(w_j, x):
    # where w_j is the weight corresponding with hidden unit j
    # x is a feature vector
    d = len(x)
    bias = w_j[3] * x[3]
    q = np.multiply(w_j, x)
    sum = np.sum(q)
    sum = sum - bias
    return sum

def applywb(w_j, x):
    # where w_j is the weight corresponding with hidden unit j
    # x is a feature vector
    d = len(x)
    #bias = w_j[0] * x[0]
    q = np.multiply(w_j, x)
    sum = np.sum(q)
    #sum = sum - bias
    return sum

def applyws(w, x):
    ones = np.ones((x.shape[0], 1))
    xb = np.append(x, ones, axis=1)
    xt = np.transpose(xb)
    dt = np.dot(w, xt)
    return np.transpose(dt)

def applywsb(w, x):
        return np.transpose(np.dot(w, np.transpose(x)))

def encodestr(names):
    d = dict([(y,x+1) for x,y in names])
    a = [d[x] for x in names]
    return a


# 1 of K y encoding
def encode(y_1, k):
    i = len(y_1)
    y_1 = y_1.astype(int)
    Y = np.zeros((i, k))
    nl = np.arange(i)
    Y[(nl), (y_1)] = 1
    return Y

def applog(z):
    return (np.log(z))

def logz(z):
    s = applog(z)
    return s

def applogm(z):
    return (np.log(1 - z))

def logm(z):
    s = applogm(z)
    return s

def appsuby(z):
    return (1 - z)

def suby(z):
    s = appsuby(z)
    return s

def looper(arr_1, arr_2, n, d):
    retarr = np.zeros((n, d))
    for i in range(0, n):
        for q in range(0, d):
            retarr[i][q] = arr_1[i] * arr_2[q]
    return retarr

def grad_W2(z, delta, wl):
    n = delta.shape[0]
    vec = (np.matmul(np.transpose(delta), z))
    sum_v = vec + wl
    return ((1 / n) * (sum_v))

def grad_W1(delta, z_mul, w_b, n_x, data, wl1):
    sum_m1 = np.dot((delta), w_b)
    z_mul = z_mul * sum_m1
    z_mul = np.dot(np.transpose(z_mul), n_x)
    z_mul = z_mul + wl1
    return ((1 / data) * z_mul)

def regwsum(W1, W2):
    w1 = W1 ** 2
    w2 = W2 ** 2
    sumw = w1.sum() + w2.sum()
    return sumw

def feedforward(W1, W2, data):
    a = applyws(W1, data)
    z = sigmoid(a)
    ones = np.ones((z.shape[0],1))
    z = np.append(z,ones,axis=1)
    b = applywsb(W2, z)
    o = sigmoid(b)
    return o

# log like error function for each input data
def errfuncsig(o, y, n):
    # y_i is the vector of 1 of k encoded y matrix at row i
    logT = logz(o)
    logM = logm(o)
    ym = suby(y)
    m_1 = logT * y
    m_2 = ym * logM
    sum_m = m_1 + m_2
    a = sum_m.sum()
    return (a * (-1/n))

# gradient with respect to weights of error functions
def graderror(y_i, o_i, x_i):
    theta = y_i - o_i
    nl = theta * x_i
    return nl

# errfunsum is the final summation of all of the error vals
def errfunsum(sum_v):
    n = len(sum_v)
    sum = np.sum(sum_v)
    return (1/n) * sum

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    data = train_data.shape[0]
    d = train_data.shape[1]
    # train_data.shape[0] = 60,000
    # train_data.shape[1] = 784
    k = W2.shape[0]
    lval = lambdaval * (1/(2*data))
    wl1 = W1 * lambdaval
    wl2 = W2 * lambdaval

    a = applyws(W1, train_data)
    z = sigmoid(a)
    ones = np.ones((z.shape[0],1))
    z = np.append(z,ones,axis=1)           # <--- adds bias to z
    o = feedforward(W1, W2, train_data)
    y = encode(train_label, k)
    obj_val = (lval * regwsum(W1, W2)) + (errfuncsig(o, y, data))


    # ----- FIND obj_grad (using equations (16) & (17)) -----

    # precursor vars
    n_x = np.append(train_data, (np.ones((train_data.shape[0],1))), axis=1)
    z_wb = np.delete(z, -1, axis=1)
    z_sub = suby(z_wb)
    z_mul = z_wb * z_sub
    delta = o - y
    w_b = np.delete(W2, -1, axis=1)

    # grad for W2
    z1 = grad_W2(z, delta, wl2)

    # grad for W1
    t_l = grad_W1(delta, z_mul, w_b, n_x, data, wl1)
    obj_grad = np.concatenate((t_l.flatten(), z1.flatten()), axis=0)


    return (obj_val, obj_grad)


def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels
    '''
    o = feedforward(W1, W2, data)
    row_i = np.argmax(o, axis=1)
    '''
    % Currently working with given datasets, however for testing purposes
    % if column vector is required use:
    % row_i = row_i[np.newaxis]
    % row_i = row_i.T
    '''
    return row_i
