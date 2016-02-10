# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pods
import flexible_function as ff      # GPSS kernel definitions
import GPy.kern as GPyKern          # GPy kernel definitions
import gpckernel                    # AutoGPC kernel definitions
from gpcdata import GPCData


print "\n\ngetGPyKernel:"
def getGPyKernelTest(kernel, data):
    print "\nInput:"
    print kernel.pretty_print()
    kern = gpckernel.GPCKernel(kernel, data, 1)
    k = kern.getGPyKernel()
    print "Output:"
    print k
    # print k.input_sensitivity()

# Data
rawdata = pods.datasets.crescent_data(seed=500)
X = rawdata['X']
Y = rawdata['Y']
Y[Y.flatten() == -1] = 0
data = GPCData(X, Y)
print data

# SE
k1 = ff.SqExpKernel(dimension=0, lengthscale=1.5, sf=0.5)
getGPyKernelTest(k1, data)
k2 = ff.SqExpKernel(dimension=1, lengthscale=1, sf=1.5)
getGPyKernelTest(k2, data)

# # Periodic
# k2 = ff.PeriodicKernel(dimension=2, lengthscale=5, period=4, sf=1.5)
# getGPyKernelTest(k2)

# Sum
k3 = ff.SumKernel([k1, k2])
getGPyKernelTest(k3, data)
k4 = ff.SumKernel([k1, k3])
getGPyKernelTest(k4, data)

# Product
k5 = ff.ProductKernel([k1, k2])
getGPyKernelTest(k5, data)
k6 = ff.ProductKernel([k1, k3])
getGPyKernelTest(k6, data)

# Expansion
kern = gpckernel.GPCKernel(k1, data, 1)
for k in kern.expand():
    print k.kernel

# Train non-sparse model
kern = gpckernel.GPCKernel(k3, data, 1)
kern.train()
kern.draw('gpckerneltest')
