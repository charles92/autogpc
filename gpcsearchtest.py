# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pods
from gpcdata import GPCData
from gpcsearch import GPCSearch

# 2D, default axis naming
data = pods.datasets.crescent_data(seed=500)
X = data['X']
Y = data['Y']
Y[Y.flatten() == -1] = 0
d = GPCData(X, Y)

search = GPCSearch(data=d, max_depth=5, beam_width=2)
results = search.search()
print "\n\nResults:"
for k in results:
	print k
