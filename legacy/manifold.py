from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], '',
            color=10,
            fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(None, cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

df = pd.read_csv('data/random_ip.csv.optimised.csv')

# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(df)
# plot_embedding(X_projected, "Random Projection")

plt.scatter(X_projected[:, 0], X_projected[:, 1])
plt.show()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(df)

knn_graph = kneighbors_graph(X_tsne, 30, include_self=False)
model = AgglomerativeClustering(linkage='complete',
                                        connectivity=knn_graph,
                                        n_clusters=None).fit(X_tsne)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=model.labels_)
plt.show()

logger.info("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, model.labels_))

#####

logger.info('Estimating Bandwidth')
bandwidth = estimate_bandwidth(X_tsne, quantile=0.1)
logger.info('Bandwidth estimate: %d' % bandwidth)

logger.info('Clustering')
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X_tsne)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

logger.info("Number of estimated clusters: %d" % n_clusters_)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=ms.labels_)
plt.show()
logger.info("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, ms.labels_))
