#!/usr/bin/env python
import sys
from sklearn.cluster import SpectralClustering
import logging
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics
import collections

#Configure logging
logging.basicConfig(level=logging.INFO,
format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

logger = logging.getLogger()

logger.info('Loading data')
df = pd.read_csv(sys.argv[1])

logger.info('Clustering')
for eigen_solver in ('arpack'):
    for assign_labels in ('kmeans', 'discretize'):
        for affinity in ('nearest_neighbors', 'rbf'):
            spectral = SpectralClustering(
                eigen_solver=eigen_solver,
                gamma=1.0,
                affinity=affinity,
                assign_labels=assign_labels,
                n_jobs=-1
            ).fit(df)
            labels = spectral.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            silhouette_score=metrics.silhouette_score(df, spectral.labels_)
            davies_bouldin_score=metrics.davies_bouldin_score(df, spectral.labels_)
            calinski_harabasz_score=metrics.calinski_harabasz_score(df, spectral.labels_)
            print(eigen_solver,assign_labels,affinity,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')

logger.info('Clustering')
for gamma in range(0,10,1):
    gamma=gamma/10
    spectral = SpectralClustering(
        eigen_solver='arpack',
        gamma=gamma,
        affinity='rbf',
        assign_labels='kmeans',
        n_jobs=-1
    ).fit(df)
    labels = spectral.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    silhouette_score=metrics.silhouette_score(df, spectral.labels_)
    davies_bouldin_score=metrics.davies_bouldin_score(df, spectral.labels_)
    calinski_harabasz_score=metrics.calinski_harabasz_score(df, spectral.labels_)
    print(gamma,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')


spectral = SpectralClustering(
    eigen_solver='arpack',
    gamma=0.4,
    affinity='rbf',
    assign_labels='kmeans',
    n_jobs=-1
).fit(df)
labels = spectral.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
silhouette_score=metrics.silhouette_score(df, spectral.labels_)
davies_bouldin_score=metrics.davies_bouldin_score(df, spectral.labels_)
calinski_harabasz_score=metrics.calinski_harabasz_score(df, spectral.labels_)
print(gamma,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')

collections.Counter(spectral.labels_)

spectral = SpectralClustering(
    eigen_solver='arpack',
    gamma=0.1,
    affinity='rbf',
    assign_labels='kmeans',
    n_jobs=-1
).fit(df)
labels = spectral.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
silhouette_score=metrics.silhouette_score(df, spectral.labels_)
davies_bouldin_score=metrics.davies_bouldin_score(df, spectral.labels_)
calinski_harabasz_score=metrics.calinski_harabasz_score(df, spectral.labels_)
print(gamma,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')

collections.Counter(spectral.labels_)

spectral = SpectralClustering(
    eigen_solver='arpack',
    gamma=1,
    affinity='rbf',
    assign_labels='kmeans',
    n_jobs=-1
).fit(df)
labels = spectral.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
silhouette_score=metrics.silhouette_score(df, spectral.labels_)
davies_bouldin_score=metrics.davies_bouldin_score(df, spectral.labels_)
calinski_harabasz_score=metrics.calinski_harabasz_score(df, spectral.labels_)
print(gamma,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')

collections.Counter(spectral.labels_)


# print('Spectral')
# print(collections.Counter(spectral.labels_))
# print(metrics.silhouette_score(df, spectral.labels_))

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(df)
# plot_2d_data(principalComponents, spectral.labels_) 

# plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c = spectral.labels_)
# plt.show()

# print(spectral)
