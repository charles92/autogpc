# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pods
from gpcdata import GPCData
from gpcsearch import GPCSearch

# 1D
print "\n=====\n1D test:"
X = np.random.uniform(1, 10, (200,1))
Y = np.zeros((200,1))
Y[150:,:] = 1
d = GPCData(X, Y)
print d

search = GPCSearch(data=d, max_depth=5, beam_width=2)
best, best1d = search.search()
print "\n=====\nBaseline:"
ck = search.baseline()
print ck
print ck.getGPyKernel()
ck.draw('./imgs/gpcsearchtest', active_dims_only=True)
print "cverror = {}".format(ck.error())
print "error = {}".format(ck.misclassifiedPoints()['X'].shape[0] / float(ck.data.X.shape[0]))

# 2D
print "\n=====\n2D test:"
data = pods.datasets.crescent_data(seed=496)
X = data['X']
Y = data['Y']
Y[Y == -1] = 0
d = GPCData(X, Y)
print d

search = GPCSearch(data=d, max_depth=5, beam_width=2)
best, best1d = search.search()
print "\n=====\nBaseline:"
ck = search.baseline()
print ck
print ck.getGPyKernel()
print "cverror = {}".format(ck.error())
print "error = {}".format(ck.misclassifiedPoints()['X'].shape[0] / float(ck.data.X.shape[0]))
