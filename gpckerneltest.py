# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import flexible_function as ff      # GPSS kernel definitions
import GPy.kern as GPyKern          # GPy kernel definitions
import gpckernel		 			# AutoGPC kernel definitions

print "\n\ngpss2gpy:"
def gpss2gpyTest(kernel):
	print "\nInput:"
	print kernel.pretty_print()
	print "Output:"
	print gpckernel.gpss2gpy(kernel)

# SE
k1 = ff.SqExpKernel(dimension=1, lengthscale=1.5, sf=0.5)
gpss2gpyTest(k1)

# Periodic
k2 = ff.PeriodicKernel(dimension=2, lengthscale=5, period=4, sf=1.5)
gpss2gpyTest(k2)

# Sum
k3 = ff.SumKernel([k1, k2])
gpss2gpyTest(k3)
k4 = ff.SumKernel([k1, k3])
gpss2gpyTest(k4)

# Product
k5 = ff.ProductKernel([k1, k2])
gpss2gpyTest(k5)
k6 = ff.ProductKernel([k1, k3])
gpss2gpyTest(k6)
