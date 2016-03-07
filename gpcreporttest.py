# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pods
from gpcdata import GPCData
from gpcreport import GPCReport
from gpcsearch import GPCSearch

data = pods.datasets.crescent_data(seed=500)
X = data['X']
Y = data['Y']
Y[Y.flatten() == -1] = 0
d = GPCData(X, Y)
print "Data size = %d" % (d.getDim() * d.getNum())

search = GPCSearch(data=d, max_depth=1, beam_width=2)
results = search.search()
print "NumDim = {0}".format(results[1].getGPyKernel().input_dim)
print "Active dims:"
for dim in results[1].getGPyKernel().active_dims:
	print dim

report = GPCReport(history=results)
report.export()
