import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Loading data from pickle file
def load_mnist_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    X, Y = data['X'], data['Y']
    return X, Y

# Applying PCA for dimensionality reduction
def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

# Applying t-SNE for dimensionality reduction
def apply_tsne(X, n_components=2):          #as we want to reduce it in two dimensions hence I have taken n_components as 2
    tsne = TSNE(n_components=n_components)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def visualize_data(X, Y, title, ax):
    for label in range(10):
        indices = np.where(Y == label)
        ax.scatter(X[indices, 0], X[indices, 1], label=str(label),s=3)
    ax.set_title(title)
    ax.legend()

# Load data
X, Y = load_mnist_data('data/mnist_small.pkl')
print("data loaded successfully")
print("applying PCA")
print("please wait. As it took around 1 minute 30 sec to execute in my laptop")
X_pca = apply_pca(X)        # Applying PCA

X_tsne = apply_tsne(X)  # Applying t-SNE

#just to represent both plots in one image using two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

visualize_data(X_pca, Y, 'PCA', ax1)
visualize_data(X_tsne, Y, 't-SNE', ax2)

plt.show()





