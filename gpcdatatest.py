# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pods
from gpcdata import GPCData

# 2D, default axis naming
data = pods.datasets.crescent_data(seed=498)
X = data['X']
Y = data['Y']
Y[Y.flatten() == -1] = 0
d = GPCData(X, Y)
print d
print d.getDataShape()
# print d.getClass(0)
# print d.getClass(1)

# 4D - Iris dataset
data = pods.datasets.iris()
X = data['X']
Y = data['Y']
versi_ind = np.where(Y == 'Iris-versicolor')
virgi_ind = np.where(Y == 'Iris-virginica')
X = np.hstack((X[versi_ind,:], X[virgi_ind,:])).squeeze()
Ynum = np.zeros(Y.size)
Ynum[virgi_ind] = 1
Ynum = np.hstack((Ynum[versi_ind], Ynum[virgi_ind])).reshape(X.shape[0], 1)
d = GPCData(X, Ynum, XLabel=('Sepal length', 'Sepal width', 'Petal length', 'Petal width'), YLabel=['Versicolor', 'Virginica'])
print d
# xx, yy, xt, yt = d.kFoldSplits()
# for i in range(5):
#   print 'split %d:' % (i+1)
#   print 'Train: %d points' % (xx[i].shape[0])
#   print np.hstack((xx[i], yy[i]))
#   print 'Test: %d points' % (xt[i].shape[0])
#   print np.hstack((xt[i], yt[i]))


# List datasets from pods:
print "\n=====\npods datasets:"
print "\n=====\nPima:"
data = pods.datasets.pima()
X, Y, XLabel = data['X'], data['Y'], data['XLabel']
d = GPCData(X, Y, XLabel=XLabel)
print d

print "\n=====\nBreast:"
data = pods.datasets.breastoriginal()
X, Y, XLabel = data['X'], data['Y'], data['XLabel']
d = GPCData(X, Y, XLabel=XLabel)
print d

print "\n=====\nBUPA:"
data = pods.datasets.bupa()
X, Y, XLabel = data['X'], data['Y'], data['XLabel']
d = GPCData(X, Y, XLabel=XLabel)
print d

print "\n=====\nCleveland:"
data = pods.datasets.cleveland()
X, Y, XLabel = data['X'], data['Y'], data['XLabel']
d = GPCData(X, Y, XLabel=XLabel)
print d
