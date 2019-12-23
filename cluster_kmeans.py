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
import collections

#Configure logging
logging.basicConfig(level=logging.INFO,
format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler()
    ])

logger = logging.getLogger()

logger.info('Loading data')
# df = pd.read_csv('data/ip2.nmap.online.hosts.masscan.csv.optimised.csv')
df = pd.read_csv(sys.argv[1])

# kmeans = KMeans(n_clusters=2, random_state=0).fit(df)

print('Clusters','Silhouette Score','Davies Bouldin Score','Calinski Harabasz Score',sep=',')

distortions = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

kmeanModel=None

K=range(0,110,10) #K=range(2,11,1)
for k in K:
    if k==0:
        k=2
    kmeanModel = KMeans(n_clusters=k, random_state=0).fit(df)
    #silhouette_score
    silhouette_score=metrics.silhouette_score(df, kmeanModel.labels_)
    silhouette_scores.append(silhouette_score)
    #davies_bouldin_score
    davies_bouldin_score=metrics.davies_bouldin_score(df, kmeanModel.labels_)
    davies_bouldin_scores.append(davies_bouldin_score)
    #calinski_harabasz_score
    calinski_harabasz_score=metrics.calinski_harabasz_score(df, kmeanModel.labels_)
    calinski_harabasz_scores.append(calinski_harabasz_score)
    print(k,silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')
    distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
    # df['class']=kmeanModel.labels_
    # df['class'].value_counts(sort=True, ascending=False)
    collections.Counter(kmeanModel.labels_).most_common(150)

#k=6
kmeanModel = KMeans(n_clusters=6, random_state=0,n_jobs=-1).fit(df)
silhouette_score=metrics.silhouette_score(df, kmeanModel.labels_)
davies_bouldin_score=metrics.davies_bouldin_score(df, kmeanModel.labels_)
calinski_harabasz_score=metrics.calinski_harabasz_score(df, kmeanModel.labels_)
print(silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')
collections.Counter(kmeanModel.labels_).most_common(150)

# logger.info('Plotting the elbow')
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion score')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

# logger.info('Plotting the silhouette scores')
# plt.plot(K, silhouette_scores, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Scores for each K')
# plt.show()


#alternative visualisations
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(2,21), metric='calinski_harabasz', timings=False,locate_elbow=False
)
visualizer.fit(df)
visualizer.show()

model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(2,21), metric='silhouette', timings=False,locate_elbow=False
)
visualizer.fit(df)
visualizer.show()

model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(2,20), timings=False
)
visualizer.fit(df)
visualizer.show()

# from joblib import dump, load
# dump(kmeanModel, 'data/model.kmean.joblib')
# #kmeanModel = load('data/model.kmean.joblib')

# df['class']=kmeanModel.labels_
# df.to_csv("data/random_ip.csv.optimised.kmeans.labeled.csv")

# df['class'].value_counts(sort=True, ascending=False)
