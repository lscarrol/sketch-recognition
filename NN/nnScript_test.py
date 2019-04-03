'''
Neural Network Script Starts here
'''
from nnFunctions import *
# you may experiment with a small data set (mnist_sample.pickle) first
#filename = 'AI_quick_draw.pickle'
filename = 'mnist_all.pickle'
train_data, train_label, test_data, test_label = preprocess(filename)

#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
accuracy_train = {0:0}
accuracy_test = {0:0}


# set the number of nodes in output unit
n_class = 10

def getaccuracy(n_hidden):


    # initialize the weights into some random matrices
    initial_W1 = initializeWeights(n_input, n_hidden)
    initial_W2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_W1.flatten(), initial_W2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)


    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
    opts = {'maxiter': 50}  # Preferred value.
    #print("Iterations    :    Training Accuracy")
    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # Reshape nnParams from 1D vector into W1 and W2 matrices
    W1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    #print("training done!")

    # Test the computed parameters

    # find the accuracy on Training Dataset
    predicted_label = nnPredict(W1, W2, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
    train_a = (100 * np.mean((predicted_label == train_label)).astype(float))
    # find the accuracy on Testing Dataset
    predicted_label = nnPredict(W1, W2, test_data)
    print('\n Test set Accuracy:    ' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
    test_a = (100 * np.mean((predicted_label == train_label)).astype(float))

    return (train_a, test_a)


for i in range(14):
    h_val = i * 2
    print("ITERATION: " + str(i) + "HIDDEN LAYERS: " + str(h_val))
    train_a, test_a = getaccuracy(h_val)
    accuracy_train.update({h_val:train_a})
    accuracy_test.update({h_val:test_a})
    print("/////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    print("/////////////////////////////////////////////////////////////////////////////////////////////////////////////")
print(accuracy_train)
print(accuracy_test)
