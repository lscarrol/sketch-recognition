# sketch-recognition

Multilayer Perceptron using CG-Minimization, Tensorflow, hyperparameter optimization on the Google Quick Draw and MNIST dataset.

## Getting Started

Navigate to /NN/MLPerceptron and run python3 create_data_set.py to create the Google Dataset

### Prerequisites

```
python3, Tensorflow, numpy, pickle, scipy 
```

### Finding optimal Hyperparameters

```
Navigate to MLPerceptron/Optimization and run the two python files. You may edit the iteration arrays, but I reccomend\ 
keeping it at default. You will find two .dat files in the directory now with lists of accuracy percentages. Either use\ np.loadtxt(filename) into an array and call np.argmax on the array to find the greatest value + index OR use another maximization\ option for larger iterations.
```
## Running the tests

Input found hyperparameters to n_hidden and lambdaval in think.py and run. The file param.picke will now contain your found
parameters of hidden units, weights from input to hidden layer, weights from hidden layer to output, and the lambdaval.

## Basic Concept

Find optimal parameters for Neural Network using log loss error, first derivative gradient descent and Hessian-CG optimization.

## Tensorflow Deep Learning

Run deepnnScript.py
Regularization not implemented yet!

## Built With

* [numpy](https://github.com/numpy/numpy) - Scientific computing with Python
* [scipy](https://github.com/scipy/scipy) - Used for minimization
* [Tensorflow](https://github.com/tensorflow/tensorflow) - Deep Learning
* [MNIST](http://yann.lecun.com/exdb/mnist/) - Hand written digit database
* [quickdraw!](https://quickdraw.withgoogle.com/data) - Google quickdraw database

 

## Authors

* **Liam Carroll** - [oswald](https://github.com/lscarrol)
* **Jeremy Baumann** - [jmb](https://github.com/jmbaumann)

See also the list of [contributors](https://github.com/lscarrol/sketch-recognition/graphs/contributors) who participated in this project.



