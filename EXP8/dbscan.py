import pandas as pd
import numpy as np
df = pd.read_csv("https://reneshbedre.github.io/assets/posts/tsne/tsne_scores.csv")

from sklearn.neighbors import NearestNeighbors
# n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
nbrs = NearestNeighbors(n_neighbors=5).fit(df)
# Find the k-neighbors of a point
neigh_dist, neigh_ind = nbrs.kneighbors(df)
# sort the neighbor distances (lengths to points) in ascending order
# axis = 0 represents sort along first axis i.e. sort along row
sort_neigh_dist = np.sort(neigh_dist, axis=0)
import matplotlib.pyplot as plt
k_dist = sort_neigh_dist[:, 4]
plt.plot(k_dist)
plt.axhline(y=2.5, linewidth=1, linestyle='dashed', color='k')
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")
plt.show()
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=2.5, min_samples=4).fit(df)
# get cluster labels
clusters.labels_


# check unique clusters
set(clusters.labels_)
from collections import Counter
Counter(clusters.labels_)
import seaborn as sns
import matplotlib.pyplot as plt
p = sns.scatterplot(data=df, x="t-SNE-1", y="t-SNE-2", hue=clusters.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
plt.show()  