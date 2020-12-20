import numpy as np
import random
from random import gauss
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


def data_generator(P, N):
    data = np.array([[0 for i in range(N+1)] for j in range(P)], dtype = float)
    for i in range(P):
        if i < P/2:
            entry = np.random.normal(0, 1, size=N+1)
            entry[N] = -1
            data[i] = entry
        else:
            entry = np.random.normal(3, 1, size=N+1)
            entry[N] = 1
            data[i] = entry
    return data

def plot_data(data):
    labels = data[:,2].astype(int)
    # Create the figure and axes objects
    fig, ax = plt.subplots(1, figsize=(10, 6))
    fig.suptitle('Example Of Labelled Scatterpoints')

    colormap = np.array(['r', 'b'])

    # Plot the scatter points
    ax.scatter(data[:,0], data[:,1],
            c = colormap[labels],  # Color of the dots
            s=100,         # Size of the dots
            alpha=0.5,     # Alpha of the dots
            linewidths=1)

    plt.show()

class RS_perceptron:
  # Constructor
    def __init__(self, nmax, learning_rate = 0.1):
        self.nmax = nmax
        self.learning_rate = learning_rate
        self.desout = 0
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
# 1 2 3 4 5 6 7 8 9 10  9  8  7  6  5  4  3  2  1
    # Train perceptron
    def train(self, X, D):
    # Initialize weights vector with zeroes
        num_features = X.shape[1]
        self.w = np.zeros(num_features + 1)
        # Perform the epochs
        for b in range(self.nmax):
            # For every combination of (X_i, D_i)
            j = 2
            for i in range(len(X) + len(X)-1):
                if i <= len(X)-1:
                    sample = X[i, :]
                    desired_outcome = D[i]
                else:
                    sample = X[i-j, :]
                    desired_outcome = D[i-j]
                    j += 2
                self.desout = desired_outcome

            # for sample, desired_outcome in zip(X, D):
                # Generate prediction and compare with desired outcome
                prediction    = self.predict(sample)
                # difference    = (desired_outcome - prediction)
                self.w[1:] += np.dot((1/num_features) * prediction * desired_outcome, sample)
                # Compute weight update via Perceptron Learning Rule
                # weight_update = self.learning_rate * difference
                # self.w[1:]    += weight_update * sample
                # self.w[0]     += weight_update
        return self

    # Generate prediction
    def predict(self, sample):
        outcome = np.dot(sample, self.w[1:]*self.desout)
        return np.where(outcome > 0, 0, 1)

def run_simulation(N = 20, alpha = 0.75, nmax = 100):
    P = int(N * alpha)
    assert (P >= 1), "P should be at least 1"
    data = data_generator(P, N)
    X = data[:,0:N]
    D = data[:,N]
    rbp = RS_perceptron(nmax)
    model = rbp.train(X, D)
    # plot_decision_regions(X, D.astype(np.integer), clf=model)
    prediction = model.predict(X)

    correct = 0
    for i in range(0, len(prediction)):
        if (prediction[i]) == 1 and (D[i] == 1):
            correct += 1
        if (prediction[i]) == 0 and (D[i] == -1):
            correct += 1
    return correct == P


ND_values = [50]
N_values = [5, 20, 50, 100, 150, 250, 500]
alpha_values = np.arange(start=0.75, stop=5.25, step=.25)
nmax_values = [100]

for N in N_values:
    for alpha in alpha_values:
        for nmax in nmax_values:
            for ND in ND_values:
                successfull_runs = 0
                for i in range(0, ND):
                    succes = run_simulation(N, alpha, nmax)
                    if succes: 
                        successfull_runs += 1
                print("N: {}, alpha: {}, nmax: {}, ND: {}, Succesfull runs: {}".format(N, alpha, nmax, ND, successfull_runs))