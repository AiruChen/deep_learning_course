import numpy as np

def sigmoid(z):
    """
    This function calculates sigmoid of z.

    Argument:
    z -- A scalar or numpy array of any size

    Returns:
    s -- sigmoid(z)
    """
    s = 1/(1 + np.exp(-z))
    return s

def initialize_with_zero(dim):
    """
    This function creates a vector of zero of shape (dim,1) for w and initialize b to 0

    Argument:
    dim -- size of w we want

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar of b
    """
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim,1))  # check if the shape is correct
    assert(isinstance(b, float) or isinstance(b, int)) # check b

    return w, b

def propagate(w, b, X, Y):
    """
    This function implements the cost function and its gradient for the propagation.

    Argument:
    w -- weight, a numpy array of size (px_height*px_width*3, 1)
    b -- bias, a scalar
    X -- training data of size(px_height*px_width*3, number of example)
    Y -- true label vector (0:noncat, 1:cat) of size(1, number of example)

    Returns:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w
    db -- gradient of the loss with respect to b
    """

    m = X.shape[1] # number of example

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = - np.sum( Y*np.log(A) + (1 - Y)*np.log(1 - A)) / m
    # Backward propagation
    dw = np.dot(X, (A - Y).T)/m
    db = np.sum(A - Y)/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw, "db":db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimize w and b by running a gradient descent algorithm

    Arguments:
    w -- weight
    b -- bias
    X -- traing data
    Y -- label vector
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- Ture to print the loss every 100 steps

    Returns:
    params -- dictionary containing w and b
    grads -- dictionary containing the gradients of the weight and bias with respect to the cost function
    costs -- list of aall the costs computed during the optimization. This will be used to plot the learning curve

    Tips:
    1) Calculate the cost and the gradient for the current parameters. Use propagate()
    2) Update the parameters using gradient descent rule for w and b
    """

    costs = []
    for i in range(num_iterations):
        # cost and gradient
        grads, cost = propagate(w, b, X, Y)

        # retrive derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update parameter
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # record the costs
        if i % 100 == 0:
            costs.append(cost)

        #print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i, %f" %(i, cost))

    params = {"w":w, "b":b}
    grads = {"dw":dw, "db":db}

    return params, grads, costs

def predict(w, b, X):
    """
    This function predicts whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weight
    b -- bias
    X -- data

    Returns:
    Y_prediction -- a numpy array containing all predictions (1/0) for X
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction = np.round(A) # A>=0.5 -> 1 ; A<0.5 -> 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    This function builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set
    Y_train -- training labels
    X_test -- test set
    Y_test -- test labels
    num_iterations -- number of iterations
    learning_rate -- learning_rate
    print_cost -- set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model
    """
    # initialize parameters with zeros
    w, b = initialize_with_zero(X_train.shape[0])
    # gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # retrieve parameters w and b
    w = parameters["w"]
    b = parameters["b"]
    # predict test/ train set
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations}

    return d


