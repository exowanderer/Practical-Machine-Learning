import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

import numpy as np
from sklearn.cluster import KMeans

# X   = np.array([[1,2],
#                 [1.5, 1.8],
#                 [5, 8],
#                 [8,8],
#                 [1,0.6],
#                 [9,11]])

import random
width   = 5.0
nPts    = 1e3
X       = np.random.normal(0,1,(nPts,2))
for k in range(X.shape[0]):
    X[k,0]  += random.randint(-1, 1)*width
    X[k,1]  += random.randint(-1, 1)*width

plt.scatter(X.T[0], X.T[1], linewidth=0)

# X_train, X_test, y_train, y_test    = cross_validation.train_test_split(X, test_size=splitRatio, random_state=random_state)
clf = KMeans(n_clusters = 9)
clf.fit(X)

centroids   = clf.cluster_centers_
labels      = clf.labels_

colors      = 10*plt.rcParams['axes.color_cycle']

for k in range(X.shape[0]):
    plt.scatter(X[k][0], X[k][1], colors[labels[k]], markersize=10, linewidth=0)

for k in range(centroids.shape[0]):
    plt.scatter(centroids[k,0], centroids[k,1], 'o', color=colors[labels[k]],  ms=50, alpha=0.5, linewidth=0)

plt.show()

'''
### KMeans on an Image: X,Y,F ###
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

import numpy as np
from sklearn.cluster import KMeans

nClusters = 100 # total guess
clf = KMeans(n_clusters = nClusters)

X,Y = np.meshgrid(np.arange(k1255_0.data.shape[0]),np.arange(k1255_0.data.shape[1]))

X_train = transpose([X.ravel(),Y.ravel(),k1255_0.data.ravel()])

clf.fit(X_train)

#%timeit clf.fit(X_train)

