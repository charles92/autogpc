# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import flexible_function as ff      # GPSS kernel definitions
import GPy.kern as GPyKern          # GPy kernel definitions


class GPCKernel(object):
    """
    GP classification kernel for AutoGPC.

    Each instance of GPCKernel is a node in the search tree. GPCKernel is a
    wrapper around the Kernel class defined in gpss-research [1]. The Kernel
    instance in GPCKernel is `translated' to a form compatible with GPy [2] and
    is subsequently used for training in GPy.

    For the time being, we support:
    1) 1-D squared exponential kernels
    2) 1-D periodic kernels
    3) sum of kernels
    4) product of kernels

    References:

    [1] The GPy authors. "GPy: A Gaussian process framework in python,"
    2012-2016.
    https://github.com/SheffieldML/GPy

    [2] J. R. Lloyd, D. Duvenaud, R. Grosse, J. B. Tenenbaum, and Z. Ghahramani,
    "Automatic construction and natural-language description of nonparametric
    regression models,"
    in Proceedings of the 28th AAAI Conference on Artificial Intelligence,
    pp. 1242-1250, June 2014.
    https://github.com/jamesrobertlloyd/gpss-research
    """

    def __init__(self, gpssKernel):
        self.kernel = gpssKernel

    def expand(self):
        kernels = []
        return kernels

    def train(self):
        return


def gpss2gpy(gpssKernel):
    """
    Convert a GPSS kernel to a GPy kernel.

    Support only:
    1) 1-D squared exponential kernels
    2) 1-D periodic kernels
    3) sum of kernels
    4) product of kernels
    """

    if isinstance(gpssKernel, ff.SqExpKernel):
        ndim = 1
        sf2 = gpssKernel.sf ** 2
        ls = gpssKernel.lengthscale
        dims = np.array([gpssKernel.dimension])
        return GPyKern.RBF(ndim, variance=sf2, lengthscale=ls, active_dims=dims)

    elif isinstance(gpssKernel, ff.PeriodicKernel):
        ndim = 1
        sf2 = gpssKernel.sf ** 2
        wl = gpssKernel.period
        ls = gpssKernel.lengthscale
        dims = np.array([gpssKernel.dimension])
        return GPyKern.StdPeriodic(ndim, variance=sf2, wavelength=wl, lengthscale=ls, active_dims=dims)

    elif isinstance(gpssKernel, ff.SumKernel):
        return GPyKern.Add(map(gpss2gpy, gpssKernel.operands))

    elif isinstance(gpssKernel, ff.ProductKernel):
        return GPyKern.Prod(map(gpss2gpy, gpssKernel.operands))

    else:
        raise NotImplementedError("Cannot translate kernel of type " + type(gpssKernel).__name__)
