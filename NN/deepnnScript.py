from __future__ import print_function
import numpy as np
import tensorflow as tf
import random as rd
import time
import pickle
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Preparing the data set
with open('AI_quick_draw.pickle', 'rb') as open_ai_quick:
    data_train = pickle.load(open_ai_quick)
    label_train1 = pickle.load(open_ai_quick)
    data_test = pickle.load(open_ai_quick)
    label_test1 = pickle.load(open_ai_quick)
n_classes = len(np.unique(label_train1))
# convert labels to 0-1 hot encoding
label_train = np.zeros((label_train1.shape[0], n_classes))
a = np.arange(label_train1.shape[0], dtype=np.int64)
b = np.array(label_train1, dtype=np.int64).reshape((label_train1.shape[0],))
label_train[a, b] = 1

label_test = np.zeros((label_test1.shape[0], n_classes))
c = np.arange(label_test1.shape[0], dtype=np.int64)
d = np.array(label_test1, dtype=np.int64).reshape((label_test1.shape[0],))
label_test[c, d] = 1


begin_time = time.time()

# Network parameters
learning_rate = 0.002
batch_size = 200
training_iteration_num = 3000

n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = data_train.shape[1]  # MNIST data input (img shape: 28*28)
n_classes = label_train.shape[1]  # MNIST total classes (0-9 digits)
dropout = 0.5

# Deep ANN model
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden fully connected layer with 128 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # layer_2 = tf.nn.dropout(layer_2, dropout)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    #with tf.device('/gpu:1'):
    sess.run(init)

    # Training cycle
    total_batch_train = int(data_train.shape[0] / batch_size)
    total_batch_test = int(data_test.shape[0] / batch_size)

    for iter_num in range(training_iteration_num):
        avg_cost_test = 0.
        avg_acc_test = 0.
        select=rd.sample(range(data_train.shape[0]), batch_size)
        train_x = data_train[select,:]
        train_y = label_train[select,:]

        _, c_train, _logits_train = sess.run([train_op, loss_op, logits], feed_dict={X: train_x, Y: train_y})

        _label_train = [np.argmax(i) for i in _logits_train]
        _label_train_y = [np.argmax(i) for i in train_y]
        _accuracy_train = np.mean(np.array(_label_train) == np.array(_label_train_y))


        for i in range(total_batch_test):
            test_x = data_test[(i) * batch_size: (i + 1) * batch_size, :]
            test_y = label_test[(i) * batch_size: (i + 1) * batch_size, :]
            c_test, _logits_test = sess.run([loss_op, logits], feed_dict={X: test_x, Y: test_y})
            avg_cost_test += c_test
            _label_test = [np.argmax(i) for i in _logits_test]
            _label_test_y = [np.argmax(i) for i in test_y]
            _accuracy_test = np.mean(np.array(_label_test) == np.array(_label_test_y))
            avg_acc_test += _accuracy_test
        if iter_num % 5 == 0:
            print("[%5d/%d] train_loss: %.5f, train_accuracy: %.5f ; test_loss: %.5f, test_accuracy: %.5f" %
                  (iter_num, training_iteration_num, c_train, _accuracy_train, avg_cost_test/total_batch_test, avg_acc_test/total_batch_test))
            #print("Test Loss: %f, Test_acc: %f" % (avg_cost_test/total_batch_test, avg_acc_test/total_batch_test))

end_time = time.time()
complete = end_time - begin_time
min = int(complete/60)
secs = round(complete % 60, 2)

print("Your program finished in %d minutes %d seconds!" % (min, secs))
