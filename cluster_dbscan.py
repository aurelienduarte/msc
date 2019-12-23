#!/usr/bin/env python
import sys
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import logging
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#Configure logging
logging.basicConfig(level=logging.INFO,
format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

logger = logging.getLogger()

logger.info('Loading data')
df = pd.read_csv(sys.argv[1])

print('eps','n_clusters_','n_noise','silhouette_score','davies_bouldin_score','calinski_harabasz_score',sep=',')
EPS=range(10,21,1)
for eps in EPS:
    eps=eps/10
    logger.info('Clustering with eps= %0.2f' % eps)
    db = DBSCAN(eps=eps, min_samples=5,n_jobs=-1).fit(df)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    silhouette_score=metrics.silhouette_score(df, db.labels_)
    davies_bouldin_score=metrics.davies_bouldin_score(df, db.labels_)
    calinski_harabasz_score=metrics.calinski_harabasz_score(df, db.labels_)
    print(eps,n_clusters_,n_noise_,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')
