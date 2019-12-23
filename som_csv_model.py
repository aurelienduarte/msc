#!/usr/bin/env python
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

df2 = pd.read_csv('data/ip2.nmap.online.hosts.masscan.csv.optimised.som_win_map8.csv')

df2_data=df2.drop(columns=['cluster'])
df2_labels=df2['cluster'].values

# silhouette_score=metrics.silhouette_score(df2_data, df2_labels)
# davies_bouldin_score=metrics.davies_bouldin_score(df2_data, df2_labels)
# calinski_harabasz_score=metrics.calinski_harabasz_score(df2_data, df2_labels)
# print(silhouette_score,davies_bouldin_score,calinski_harabasz_score,sep=',')
# collections.Counter(df2_labels)

sns.pairplot(df2, hue='cluster')
plt.show()
