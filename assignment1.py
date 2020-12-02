import numpy as np 
import random
from random import gauss
import matplotlib.pyplot as plt

def data_generator(P):
    data = np.array([[0 for i in range(3)] for j in range(P)], dtype = float)
    for i in range(P):
        data[i,0] = gauss(0,1)
        data[i,1] = gauss(0,1)
        data[i,2] = random.choice([0,1])
    return data

data = data_generator(10)
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
