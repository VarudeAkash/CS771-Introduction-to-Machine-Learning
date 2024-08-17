from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import matplotlib
# Function for plotting data
def plot_data(data, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c='blue')
    plt.xlabel('axis 1')
    plt.ylabel('axis 2')
    plt.title(title)
    plt.show()

# Function for plotting K-means clustering results
def plot_kmeans_clusters(data, clusters, title):
    colors = ['red', 'green']
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('axis 1')
    plt.ylabel('axis 2')
    plt.title(title)
    plt.show()

# Load data
kmdata = np.loadtxt(fname='data/kmeans_data.txt', dtype=float, delimiter=None)
print("data loaded successfully")
# Plot original data
plot_data(kmdata, 'Our Data in its original space')

# Handcrafted Feature Matrix
skmdata = kmdata ** 2

# Ploting data in the transformed space
plot_data(skmdata, 'Plotting data in transformed space')

# K-means
c1, c2 = skmdata[0, :], skmdata[1, :]
skmdatadf = pd.DataFrame(skmdata)
z = np.zeros(skmdata.shape[0])
skmdatadf[2] = z
d1, d2 = skmdata[0, :], skmdata[1, :]
c1 = [0, 0]
c2 = [0, 0]

while not (np.array_equal(c1, d1)) or not (np.array_equal(c2, d2)):
    c1 = d1  # initializing cluster 1 center as per given in question
    c2 = d2  # initializing cluster 2 center as per given in question
    for i in range(0, skmdata.shape[0]):            #assigning which cluster each data point belong to depending on eucleadian distance
        if distance.euclidean(skmdata[i, :], c1) < distance.euclidean(skmdata[i, :], c2):
            skmdatadf.loc[i, 2] = 1
        else:
            skmdatadf.loc[i, 2] = 2
    # computing new centers
    mn = skmdatadf.groupby([2]).mean()
    d1 = list(mn.loc[1, :])
    d2 = list(mn.loc[2, :])

# Ploting the results after running K-means algorithm
plot_kmeans_clusters(skmdata, skmdatadf[2], 'Clustering results in transformed feature space after running Kmeans')

# Plotting the data in the original feature space
plot_kmeans_clusters(kmdata, skmdatadf[2], 'Clustering results in original feature space after running Kmeans')
