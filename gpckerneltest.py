# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import flexible_function as ff      # GPSS kernel definitions
import GPy.kern as GPyKern          # GPy kernel definitions
import gpckernel                    # AutoGPC kernel definitions

print "\n\ngetGPyKernel:"
def getGPyKernelTest(kernel):
    print "\nInput:"
    print kernel.pretty_print()
    kern = gpckernel.GPCKernel(kernel, 3, 1)
    k = kern.getGPyKernel()
    print "Output:"
    print k
    # print k.input_sensitivity()

# SE
k1 = ff.SqExpKernel(dimension=0, lengthscale=1.5, sf=0.5)
getGPyKernelTest(k1)
k2 = ff.SqExpKernel(dimension=1, lengthscale=1, sf=1.5)
getGPyKernelTest(k2)

# # Periodic
# k2 = ff.PeriodicKernel(dimension=2, lengthscale=5, period=4, sf=1.5)
# getGPyKernelTest(k2)

# Sum
k3 = ff.SumKernel([k1, k2])
getGPyKernelTest(k3)
k4 = ff.SumKernel([k1, k3])
getGPyKernelTest(k4)

# Product
k5 = ff.ProductKernel([k1, k2])
getGPyKernelTest(k5)
k6 = ff.ProductKernel([k1, k3])
getGPyKernelTest(k6)

# Expansion
kern = gpckernel.GPCKernel(k1, 3, 1)
for k in kern.expand():
    print k.kernel
