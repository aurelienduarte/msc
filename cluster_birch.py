#!/usr/bin/env python
import sys
from sklearn.cluster import Birch
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


print('threshold','n_clusters_','n_noise','silhouette_score','davies_bouldin_score','calinski_harabasz_score',sep=',')

logger.info('Generating clusters')
K = range(5,11,1)
for k in K:
    threshold=k/10
    print('threshold',threshold)
    brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold,compute_labels=True)
    brc.fit(df)
    # brc.predict(df)
    labels = brc.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    silhouette_score=metrics.silhouette_score(df, labels)
    davies_bouldin_score=metrics.davies_bouldin_score(df, labels)
    calinski_harabasz_score=metrics.calinski_harabasz_score(df, labels)
    print(threshold,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')
