#!/usr/bin/env python
import sys
from sklearn.cluster import KMeans
import logging
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics

#Configure logging
logging.basicConfig(level=logging.INFO,
format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

logger = logging.getLogger()

logger.info('Loading data')
df = pd.read_csv(sys.argv[1])

# kmeans = KMeans(n_clusters=2, random_state=0).fit(df)

n_samples=len(df.index)

logger.info('Generating clusters')
distortions = []
K = range(2,20,1)
for k in K:
    logger.info('KMeans Clusters='+str(k))
    kmeanModel = KMeans(n_clusters=k, random_state=0).fit(df)
    distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
    logger.info("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, kmeanModel.labels_))
    print(collections.Counter(kmeanModel.labels_).most_common(10))

logger.info('Plotting the elbow')
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
