import numpy as np 
import random
from random import gauss
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


def data_generator(P):
    data = np.array([[0 for i in range(3)] for j in range(P)], dtype = float)
    for i in range(P):
        # data[i,2] = random.choice([0,1])
        if i < P/2:
            data[i,0] = gauss(0,1)
            data[i,1] = gauss(0,1)
            data[i,2] = 0
        else:  
            data[i,0] = gauss(3,1)
            data[i,1] = gauss(3,1)
            data[i,2] = 1
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
    def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate

    # Train perceptron
    def train(self, X, D):
    # Initialize weights vector with zeroes
        num_features = X.shape[1]
        self.w = np.zeros(num_features + 1)
        # Perform the epochs
        for i in range(self.number_of_epochs):
            # For every combination of (X_i, D_i)
            for sample, desired_outcome in zip(X, D):
                # Generate prediction and compare with desired outcome
                prediction    = self.predict(sample)
                difference    = (desired_outcome - prediction)
                # Compute weight update via Perceptron Learning Rule
                weight_update = self.learning_rate * difference
                self.w[1:]    += weight_update * sample
                self.w[0]     += weight_update
        return self

    # Generate prediction
    def predict(self, sample):
        outcome = np.dot(sample, self.w[1:]) + self.w[0]
        return np.where(outcome > 0, 1, 0)

data = data_generator(10)
# plot_data(data)
X = data[:,0:2]
D = data[:,2]
rbp = RS_perceptron(600, 0.1)
model = rbp.train(X, D)
plot_decision_regions(X, D.astype(np.integer), clf=model)
plt.title('Perceptron')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()