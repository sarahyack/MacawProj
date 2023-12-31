import numpy as np

from model_functions import *

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
        
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)
        
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        da0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

def L_layer_model(X, Y, layers_dims, X_cv=None, Y_cv=None, learning_rate = 0.001, num_iterations = 3000, print_cost=False, datagen=None):
        """
        Trains an L-layer neural network model using gradient descent.

        Parameters:
        - X: Input data of shape (n_x, m) where n_x is the number of features and m is the number of examples.
        - Y: True "label" vector of shape (1, m).
        - layers_dims: List containing the number of units in each layer of the network.
        - X_cv: (Optional) Input data for cross-validation.
        - Y_cv: (Optional) True "label" vector for cross-validation.
        - learning_rate: Learning rate for gradient descent.
        - num_iterations: Number of iterations of the optimization loop.
        - print_cost: (Optional) If True, print the cost every 100 iterations.

        Returns:
        - parameters: Dictionary containing the parameters learned by the model.
        - history: Dictionary containing the cost history during training and cross-validation (if provided).
        """

        history = {
            "loss": [],
            "val_loss": [],
        }
        np.random.seed(1)
        costs = []                         # keep track of cost

        # Parameters initialization.
        parameters = initialize_parameters(layers_dims)


        if datagen is not None:
            augmented_X = []
            augmented_Y = []

            for x_batch, y_batch in datagen.flow(X, Y, batch_size=32):  # Adjust batch size as needed
                augmented_X.append(x_batch)
                augmented_Y.append(y_batch)
                if len(augmented_X) >= len(X) / 32:  # Enough augmented data
                    break

            augmented_X = np.concatenate(augmented_X)
            augmented_Y = np.concatenate(augmented_Y)

            X = np.concatenate((X, augmented_X))
            Y = np.concatenate((Y, augmented_Y))


        v, s = initialize_adam(parameters)
        t = 1
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            AL, caches = L_model_forward(X, parameters)

            # Compute cost.
            cost = compute_cost(AL, Y)
            history["loss"].append(cost)

            if X_cv is not None and Y_cv is not None:
                B_val, M_val,  _ = L_model_forward(X_cv, parameters)
                val_cost = compute_cost(B_val, M_val, Y_cv)
                history["val_loss"].append(val_cost)

            # Backward propagation.
            grads = L_model_backward(AL, Y, caches)
                
            # Update parameters.
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate)
            t+=1

            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Avg Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
            
        return parameters, history
