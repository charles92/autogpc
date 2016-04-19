# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pods
from gpcdata import GPCData
from gpcreport import GPCReport
from gpcsearch import GPCSearch

data = pods.datasets.crescent_data(seed=498)
X = data['X']
Y = data['Y']
Y[Y.flatten() == -1] = 0
d = GPCData(X, Y)
print "Data size = %d" % (d.getDim() * d.getNum())

search = GPCSearch(data=d, max_depth=2, beam_width=2)
results = search.search()

report = GPCReport(history=results)
report.export()
