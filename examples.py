# https://educative.io/courses/fundamentals-of-machine-learning-for-software-engineers
import numpy as np
import matplotlib.pyplot as plt

from classes import Dataset, Plot2D
from functions import plot_2d_data

def first_learning_program():
    # Returns the prediction (y_hat)
    # X is the vector data we're predicting from, w is the weight, and b is the bias
    def predict(X, w, b):
        return X * w + b
    
    # L2 loss function
    def loss(X, Y, w, b):
        return np.average((predict(X, w, b) - Y) ** 2)
    
    # lr is the learning rate (aka step size)
    def train(X, Y, iterations, lr):
        w = b = 0
        for i in range(iterations):
            current_loss = loss(X, Y, w, b)
            if i % 300 == 0:
                print("Iteration %4d => Loss: %.6f" % (i, current_loss))

            if loss(X, Y, w + lr, b) < current_loss: # Updating weight
                w += lr
            elif loss(X, Y, w - lr, b) < current_loss: # Updating weight
                w -= lr
            elif loss(X, Y, w, b + lr) < current_loss: # Updating bias
                b += lr
            elif loss(X, Y, w, b - lr) < current_loss: # Updating bias
                b -= lr
            else:
                return w, b

        raise Exception("Couldn't converge within %d iterations" % iterations)


    X, Y = np.loadtxt("data/pizza.txt", skiprows=1, unpack=True)  # load data

    # Train the system
    w, b = train(X, Y, iterations=10000, lr=0.01)
    print("\nw=%.3f, b=%.3f" % (w, b))

    # Predict the number of pizzas
    print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

    regression_data = [X, predict(X, w, b)]

    plot_2d_data(X, Y, "Reservations", "Pizzas", other_data=regression_data)

def gradient_descent():
    def predict(X, w, b):
        return X * w + b
    # L2 loss function
    def loss(X, Y, w, b):
        return np.average((predict(X, w, b) - Y) ** 2)
    def gradient(X, Y, w, b):
        w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
        b_gradient = 2 * np.average(predict(X, w, b) - Y)
        return (w_gradient, b_gradient)
    
    def train(X, Y, iterations, lr):
        w = b = 0
        for i in range(iterations):
            if (i % 5000 == 0):
                print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
            w_gradient, b_gradient = gradient(X, Y, w, b)
            w -= w_gradient * lr
            b -= b_gradient * lr
        return w, b
    

    X, Y = np.loadtxt("data/pizza.txt", skiprows=1, unpack=True)
    w, b = train(X, Y, iterations=20000, lr=0.001)

    regression_data = [X, predict(X, w, b)]

    plot_2d_data(X, Y, "Reservations", "Pizzas", other_data=regression_data)
    print("\nw=%.10f" % w)

def upgraded_gradient_descent():
    # computing the predictions
    def predict(X, w):
        return np.matmul(X, w)

    # calculating the loss
    def loss(X, Y, w):
        return np.average((predict(X, w) - Y) ** 2)

    # evaluating the gradient
    def gradient(X, Y, w):
        return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

    # performing the training phase for our classifier
    def train(X, Y, iterations, lr):
        w = np.zeros((X.shape[1], 1))
        for i in range(iterations):
            print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
            w -= gradient(X, Y, w) * lr
        return w
    

    # TODO - separate below out
    x1, x2, x3, y = np.loadtxt("data/pizza_3_vars.txt", skiprows=1, unpack=True)
    
    # NOTE - np.ones is used to add a bias term to the model
    X = np.column_stack((np.ones(x1.size), x1, x2, x3))
    Y = y.reshape(-1, 1)
    
    w = train(X, Y, iterations=50000, lr=0.001)
    print("\nWeights: %s" % w.T)
    print("\nA few predictions:")
    for i in range(5):
        print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))

    return w