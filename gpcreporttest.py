# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pods
from gpcdata import GPCData
from gpcreport import GPCReport
from gpcsearch import GPCSearch

data = pods.datasets.pima()
X = data['X'][:250,[1,5,6,7]]
Y = data['Y'][:250]

d = GPCData(X, Y, XLabel=['plasma glucose', 'BMI', 'pedigree function', 'age'])
print "\n=====\nData size: D = %d, N = %d." % (d.getDim(), d.getNum())

search = GPCSearch(data=d, max_depth=3, beam_width=1)
best, best1d = search.search()

print "\n=====\nBaseline:"
ck = search.baseline()
print ck
print "error = {}".format(ck.error())

report = GPCReport(name='Pima', history=best, best1d=best1d, constkernel=ck)
report.export()


# data = pods.datasets.iris()
# X = data['X']
# Y = data['Y']
# versi_ind = np.where(Y == 'Iris-versicolor')
# virgi_ind = np.where(Y == 'Iris-virginica')
# X = np.hstack((X[versi_ind,:], X[virgi_ind,:])).squeeze()
# Ynum = np.zeros(Y.size)
# Ynum[virgi_ind] = 1
# Ynum = np.hstack((Ynum[versi_ind], Ynum[virgi_ind])).reshape(X.shape[0], 1)
# d = GPCData(X, Ynum, XLabel=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'], YLabel=['Versicolor', 'Virginica'])
# print "Data size = %d" % (d.getDim() * d.getNum())
# search = GPCSearch(data=d, max_depth=4, beam_width=2)
# best, best1d = search.search()

# print "\n=====\nBaseline:"
# ck = search.baseline()
# print ck
# print "error = {}".format(ck.error())

# report = GPCReport(name='Iris', history=best, best1d=best1d, constkernel=ck)
# report.export()


# data = pods.datasets.crescent_data(seed=498)
# X = data['X']
# Y = data['Y']
# Y[Y.flatten() == -1] = 0
# X2 = X + np.random.randn(*X.shape)
