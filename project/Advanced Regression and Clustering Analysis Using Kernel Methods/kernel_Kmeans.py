from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance

# Loading data
kmdata = np.loadtxt(fname='data/kmeans_data.txt', dtype=float, delimiter=None)
print("data read successfully\n")

data_features = pd.DataFrame(np.zeros(kmdata.shape[0]))  # data_features stores feature of each data along with the cluster id
data_features[1] = np.zeros(kmdata.shape[0])
N=data_features.shape[0]        #Total number of data points


def dataPlotting(kmdata):
    # Plotting the data
    fig = plt.figure(figsize=(8, 8))
    plt.xlabel('axis 1')
    plt.ylabel('axis 2')
    colors = ['red', 'green']
    matplotlib.pyplot.plot(kmdata[r][0], kmdata[r][1], 'bo')
    matplotlib.pyplot.scatter(kmdata[:, 0], kmdata[:, 1], c=data_features[1], cmap=matplotlib.colors.ListedColormap(colors))
    plt.title('Kmeans clustering based on 1 landmark-based rbf kernel features')
    plt.show()

for i in range(0, 10):  #iterating 10 times as mentioned in the question

    #Computing the landmark features
    r = np.random.randint(0, N)
    for j in range(0, N):
        data_features.loc[j][0] = np.exp(-0.1 * ((distance.euclidean(kmdata[r, :], kmdata[j, :])) ** 2)) #using rbf kernel to get features of data points

    #Implementing Kmeans algorithm
    d1 = data_features.loc[0, 0]
    d2 = data_features.loc[1, 0]
    c1 = 0
    c2 = 0
    while not (np.array_equal(c1, d1)) or not (np.array_equal(c2, d2)):
        c1 = d1
        c2 = d2
        for i in range(0, N):
            if distance.euclidean(np.array([data_features.loc[i, 0]]), np.array([c1])) < distance.euclidean(np.array([data_features.loc[i, 0]]), np.array([c2])):
                data_features.loc[i, 1] = 1
            else:
                data_features.loc[i, 1] = 2
        mn = data_features.groupby([1]).mean()
        # recomputing centers
        d1 = mn.loc[1, 0]
        d2 = mn.loc[2, 0]
    dataPlotting(kmdata)
    
