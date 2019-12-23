#!/usr/bin/env python
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth
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

print('quantile','n_clusters_','n_noise','silhouette_score','davies_bouldin_score','calinski_harabasz_score',sep=',')
r=range(0,11,1)
for quantile in r:
    quantile=quantile/10
    # logger.info('Estimating Bandwidth')
    bandwidth = estimate_bandwidth(df, quantile=quantile,n_jobs=-1)
    logger.info('Bandwidth estimate: %f, quantile: %f' % (bandwidth, quantile))
    if bandwidth>0.0:
        # logger.info('Clustering')
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True,n_jobs=-1).fit(df)
        labels = ms.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(n_clusters_)
        if n_clusters_>1:
            silhouette_score=metrics.silhouette_score(df, ms.labels_)
            davies_bouldin_score=metrics.davies_bouldin_score(df, ms.labels_)
            calinski_harabasz_score=metrics.calinski_harabasz_score(df, ms.labels_)
            print(quantile,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')


#Further attempts
# logger.info('Clustering')
ms = MeanShift(bin_seeding=True,n_jobs=-1).fit(df)
labels = ms.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print(n_clusters_)
if n_clusters_>1:
    silhouette_score=metrics.silhouette_score(df, ms.labels_)
    davies_bouldin_score=metrics.davies_bouldin_score(df, ms.labels_)
    calinski_harabasz_score=metrics.calinski_harabasz_score(df, ms.labels_)
    print(quantile,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')
