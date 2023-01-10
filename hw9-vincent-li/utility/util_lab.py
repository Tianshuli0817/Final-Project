import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import load_iris


def configure_plots():
    '''Configures plots by making some quality of life adjustments'''
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['lines.linewidth'] = 2

    
def load_toy(n, k, d=2, scale=1, width=0.05, random_state=None):
    '''
    Generates N points sampled from K clusters in R^d space
    '''
    if random_state:
        np.random.seed(random_state)
    
    centroids = np.random.rand(k, d) * scale
    sizes = divvy(n, k)
    data = np.concatenate([np.c_[np.random.normal(centroid, width, size=(size, d)), np.ones(size) * i]
                           for i, (centroid, size) in enumerate(zip(centroids, sizes))])

    return data[:, :d], data[:, d]


def divvy(total, num_slices):
    '''Divvies TOTAL into NUM_SLICES sizes'''
    
    size, left = total // num_slices, total % num_slices
    return [size + (left - i > 0) for i in range(num_slices)]



def plot_kmeans(X, centroids, prev_centroids=None, assignments=None):
    '''
    Creates k-means plots
    '''
    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=120, c='blue', marker='s')
    
    if prev_centroids is not None:
        plt.scatter(prev_centroids[:, 0], prev_centroids[:, 1], s=200, c='orange', marker='s')
        plt.legend(['data points', 'final centroids','inital centroids'])
    else:
        plt.legend(['data points', 'final centroids'])

    plt.title('Toy Clustering Data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
