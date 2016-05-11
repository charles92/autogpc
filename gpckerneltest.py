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

# Periodic
kp = ff.PeriodicKernel(dimension=0, lengthscale=5, period=4, sf=1.5)
getGPyKernelTest(k2, data)

# Sum
k3 = ff.SumKernel([k1, k2])
getGPyKernelTest(k3, data)
k4 = ff.SumKernel([k1, kp])
getGPyKernelTest(k4, data)

# Product
k5 = ff.ProductKernel([k1, k2])
getGPyKernelTest(k5, data)
k6 = ff.ProductKernel([k1, k3])
getGPyKernelTest(k6, data)

# gpy2gpss
print "\n\nGPy2GPSS:"
kk1 = GPyKern.RBF(1, variance=2, lengthscale=3, active_dims=np.array([3]))
kk2 = GPyKern.RBF(1, variance=4, lengthscale=6, active_dims=np.array([1]))
kks = GPyKern.Add([kk1, kk2])
print kks
print gpckernel.gpy2gpss(kks).pretty_print()

# Expansion
kern = gpckernel.GPCKernel(k1, data, 1)
print "\n\nBefore expansion:"
print k1.pretty_print()
print "\nAfter expansion:"
for k in kern.expand():
    print k.kernel.pretty_print()

# Train non-sparse model
kern = gpckernel.GPCKernel(k5, data, 1)
kern.train()
kern.draw('imgs/gpckerneltest')

# Kernel equality
print "\n\nKernel equality:"
k1 = ff.NoneKernel()
k2 = ff.NoneKernel()
assert gpckernel.isKernelEqual(k1, k2)

k2 = ff.SqExpKernel(dimension=0, lengthscale=1.5, sf=0.5)
assert not gpckernel.isKernelEqual(k1, k2)

k1 = ff.SqExpKernel(dimension=0, lengthscale=1, sf=0.5)
assert gpckernel.isKernelEqual(k1, k2)
assert not gpckernel.isKernelEqual(k1, k2, compare_params=True)
assert gpckernel.isKernelEqual(k1, k1.copy(), compare_params=True)

k2 = ff.SqExpKernel(dimension=1, lengthscale=1, sf=0.5)
assert not gpckernel.isKernelEqual(k1, k2)

k1 = ff.SqExpKernel(dimension=0, lengthscale=1, sf=0.5)
k2 = ff.SqExpKernel(dimension=1, lengthscale=1, sf=0.5)
k3 = ff.SumKernel([k1, k2])
k4 = ff.SumKernel([k2, k1])
assert gpckernel.isKernelEqual(k3, k4)
k4 = ff.SumKernel([k1, k1])
assert not gpckernel.isKernelEqual(k3, k4)

# Misclassified Points
print '\n\nMisclassified Points:'
k0 = ff.SqExpKernel(dimension=0, lengthscale=1, sf=1.5)
k1 = ff.SqExpKernel(dimension=1, lengthscale=1, sf=1)
k = gpckernel.GPCKernel(ff.SumKernel([k0, k1]), data, depth=2)
k.train()
print 'active_dims:'
print k.getActiveDims()
print k.misclassifiedPoints()['X']
print k.misclassifiedPoints()['Y']
mis = k.misclassifiedPoints()
nMis = mis['X'].shape[0]
assert nMis == mis['Y'].shape[0]
print 'X\tY\tMisclassified as'
for i in range(nMis):
    print np.array_str(mis['X'][i,:]) + '\t' + np.array_str(mis['Y'][i])

print "Passed!"
