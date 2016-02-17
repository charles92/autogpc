# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import flexible_function as ff      # GPSS kernel definitions
import grammar                      # GPSS kernel expansion
import GPy
from gpcplot import GPCPlot


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

    def __init__(self, gpssKernel, data, depth=0):
        """
        :param gpssKernel: a GPSS kernel as defined in flexible_function.py
        :param data: object of type GPCData which the kernel works on
        :param depth: depth of the current node in the search tree (root is 0)
        """
        self.kernel = gpssKernel
        self.data = data
        self.depth = depth
        self.model = None
        self.isSparse = None
        if not isinstance(self.kernel, ff.NoneKernel):
            self.kernel.initialise_params(data_shape=self.data.getDataShape())

    def __repr__(self):
        kernel_str = self.kernel.pretty_print()
        if isinstance(kernel_str, Exception):
            kernel_str = "No Expression"
        return 'GPCKernel: depth = %d, NLML = %f\n' % \
               (self.depth, self.getNLML()) + \
               kernel_str

    def equals(self, other, strict=False, canonical=True):
        """
        Check if this kernel is equivalent to another kernel.

        :param other: kernel of type GPCKernel which is to be compared
        :param strict: check the equality of hyperparameters; default to False
        :param canonical: convert the kernel to canonical form before comparison;
        default to True
        :returns: True if the two kernels are equivalent, False otherwise
        """
        return isKernelEqual(self.kernel, other.kernel, compare_params=strict, use_canonical=canonical)

    def expand(self, base_kernels='SE'):
        """
        Expand this kernel using grammar defined in grammar.py.
        :returns: list of GPCKernel resulting from the expansion
        """
        ndim = self.data.getDim()
        g = grammar.MultiDGrammar(ndim, base_kernels=base_kernels, rules=None)
        kernels = grammar.expand(self.kernel, g)
        kernels = [k.canonical() for k in kernels]
        map(lambda k: k.initialise_params(data_shape=self.data.getDataShape()), kernels)
        kernels = [k.simplified() for k in kernels]
        kernels = ff.remove_duplicates(kernels)
        kernels = [k for k in kernels if not isinstance(k, ff.NoneKernel)]
        kernels = [GPCKernel(k, self.data, self.depth + 1) for k in kernels]
        return kernels

    def train(self):
        """
        Train a GP classification model using all data points
        """
        self.model = GPy.models.GPClassification( \
            self.data.X, \
            self.data.Y, \
            kernel=self.getGPyKernel())
        self.isSparse = False
        self.model.optimize()
        self.kernel = gpy2gpss(self.model.kern)

    def trainSparse(self, num_inducing=20):
        """
        Train a sparse GP classification model using inducing points
        """
        self.model = GPy.models.SparseGPClassification( \
            self.data.X, \
            self.data.Y, \
            kernel=self.getGPyKernel(), \
            num_inducing=num_inducing)
        self.isSparse = True
        self.model.optimize()
        self.kernel = gpy2gpss(self.model.kern)

    def draw(self, filename):
        """
        Plot the model and data points
        :param file: the output file (path and) name, without extension
        """
        plot = GPCPlot.create(self.model, self.data.XLabel, usetex=True)
        plot.draw()
        plot.save(filename)

    def getDepth(self):
        """
        :returns: depth of this kernel in the search tree
        """
        return self.depth

    def getNLML(self):
        """
        :returns: negative log marginal likelihood
        """
        if self.model is not None:
            return -self.model.log_likelihood()
        else:
            return float("inf")

    def getGPyKernel(self):
        """
        Convert this GPCKernel to GPy kernel.
        :returns: an object of type GPy.kern.Kern
        """
        return gpss2gpy(self.kernel)


def gpss2gpy(kernel):
    """
    Convert a GPSS kernel to a GPy kernel recursively.

    Support only:
    1) 1-D squared exponential kernels
    2) 1-D periodic kernels
    3) sum kernels
    4) product kernels

    :param kernel: a GPSS kernel as defined in flexible_function.py
    :returns: an object of type GPy.kern.Kern
    """
    assert isinstance(kernel, ff.Kernel), "kernel must be of type flexible_function.Kernel"

    if isinstance(kernel, ff.SqExpKernel):
        ndim = 1
        sf2 = kernel.sf ** 2
        ls = kernel.lengthscale
        dims = np.array([kernel.dimension])
        return GPy.kern.RBF(ndim, variance=sf2, lengthscale=ls, active_dims=dims)

    elif isinstance(kernel, ff.PeriodicKernel):
        ndim = 1
        sf2 = kernel.sf ** 2
        wl = kernel.period
        ls = kernel.lengthscale
        dims = np.array([kernel.dimension])
        return GPy.kern.StdPeriodic(ndim, variance=sf2, wavelength=wl, lengthscale=ls, active_dims=dims)

    elif isinstance(kernel, ff.SumKernel):
        return GPy.kern.Add(map(gpss2gpy, kernel.operands))

    elif isinstance(kernel, ff.ProductKernel):
        return GPy.kern.Prod(map(gpss2gpy, kernel.operands))

    else:
        raise NotImplementedError("Cannot translate kernel of type " + type(kernel).__name__)


def gpy2gpss(kernel):
    """
    Convert a GPy kernel to a GPSS kernel recursively.

    Support only:
    1) 1-D squared exponential kernels
    2) 1-D periodic kernels
    3) sum kernels
    4) product kernels

    :param kernel: a GPSS kernel as defined in flexible_function.py
    :returns: an object of type GPy.kern.Kern
    """
    assert isinstance(kernel, GPy.kern.Kern), "kernel must be of type GPy.kern.Kern"

    if isinstance(kernel, GPy.kern.RBF):
        sf = np.sqrt(kernel.variance)[0]
        ls = kernel.lengthscale[0]
        dim = kernel.active_dims[0]
        return ff.SqExpKernel(dimension=dim, lengthscale=ls, sf=sf)

    elif isinstance(kernel, GPy.kern.StdPeriodic):
        sf = np.sqrt(kernel.variance)[0]
        ls = kernel.lengthscales[0]
        per = kernel.wavelengths[0]
        dim = kernel.active_dims[0]
        return ff.PeriodicKernel(dimension=dim, lengthscale=ls, period=per, sf=sf)

    elif isinstance(kernel, GPy.kern.Add):
        return ff.SumKernel(map(gpy2gpss, kernel.parts))

    elif isinstance(kernel, GPy.kern.Prod):
        return ff.ProductKernel(map(gpy2gpss, kernel.parts))

    else:
        raise NotImplementedError("Cannot translate kernel of type " + type(kernel).__name__)


def isKernelEqual(k1, k2, compare_params=False, use_canonical=True):
    """
    Compare two GPSS kernels recursively.

    Support only:
    1) 1-D squared exponential kernels
    2) 1-D periodic kernels
    3) sum of kernels
    4) product of kernels
    5) NoneKernel

    :param k1: GPSS kernel for comparison
    :param k2: another GPSS kernel for comparison
    :param compare_params: compare functional form only if False (default),
    otherwise compare hyperparameters as well
    :param use_canonical: convert kernels to canonical form before comparison if
    True (default)
    :returns: True if two kernels are equal, False otherwise
    """
    assert isinstance(k1, ff.Kernel), "k1 must be of type flexible_function.Kernel"
    assert isinstance(k2, ff.Kernel), "k2 must be of type flexible_function.Kernel"

    if use_canonical:
        k1 = k1.canonical()
        k2 = k2.canonical()

    if isinstance(k1, ff.NoneKernel):
        return isinstance(k2, ff.NoneKernel)

    elif isinstance(k1, ff.SqExpKernel):
        result = isinstance(k2, ff.SqExpKernel) and k1.dimension == k2.dimension
        if compare_params:
            result = result and np.array_equal(k1.param_vector, k2.param_vector)
        return result

    elif isinstance(k1, ff.PeriodicKernel):
        result = isinstance(k2, ff.PeriodicKernel) and k1.dimension == k2.dimension
        if compare_params:
            result = result and np.array_equal(k1.param_vector, k2.param_vector)
        return result

    elif isinstance(k1, ff.SumKernel):
        result = isinstance(k2, ff.SumKernel) and len(k1.operands) == len(k2.operands)
        result = result and \
            all([isKernelEqual(o1, o2, compare_params=compare_params, use_canonical=False) \
            for (o1, o2) in zip(k1.operands, k2.operands)])
        return result

    elif isinstance(k1, ff.ProductKernel):
        result = isinstance(k2, ff.ProductKernel) and len(k1.operands) == len(k2.operands)
        result = result and \
            all([isKernelEqual(o1, o2, compare_params=compare_params, use_canonical=False) \
            for (o1, o2) in zip(k1.operands, k2.operands)])
        return result

    else:
        raise NotImplementedError("Cannot compare kernels of type " \
            + type(k1).__name__ + " and " + type(k2).__name__)
