import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, x_label, y_label, filename=None):
    plt.plot(x, y, 'r')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if filename:
        plt.savefig(filename, format='png')
        plt.close()
        return
    plt.show()


def identity(x):
    y = x
    x_label = 'x'
    y_label = 'f(x) = x'
    plot(x, y, x_label, y_label, 'identity.png')


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    x_label = 'x'
    y_label = 'f(x) = σ(x)'
    plot(x, y, x_label, y_label, 'sigmoid.png')


def tanh(x):
    y = np.tanh(x)
    x_label = 'x'
    y_label = 'f(x) = tanh(x)'
    plot(x, y, x_label, y_label, 'tanh.png')


def step_function(x):
    y = np.where(x > 0, 1, -1)
    x_label = 'x'
    y_label = 'f(x) = 1 if x > 0, -1 otherwise'
    plot(x, y, x_label, y_label, 'step_function.png')


def relu(x):
    y = np.maximum(x, 0)
    x_label = 'x'
    y_label = 'f(x) = max(0, x)'
    plot(x, y, x_label, y_label, 'relu.png')


def leaky_relu(x):
    y = np.maximum(x, 0.1 * x)
    x_label = 'x'
    y_label = 'f(x) = max(0.1x, x)'
    plot(x, y, x_label, y_label, 'leaky_relu.png')


if __name__ == "__main__":
    x = np.linspace(-10, 10, 100000)
    identity(x)
    sigmoid(x)
    tanh(x)
    step_function(x)
    relu(x)
    leaky_relu(x)
