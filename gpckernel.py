# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import itertools as it
import numpy as np
import flexible_function as ff      # GPSS kernel definitions
import grammar                      # GPSS kernel expansion
import GPy
from gpcplot import GPCPlot
from gpcdata import GPCData


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

    def __init__(self, gpssKernel, data, depth=0, parent=None):
        """
        :param gpssKernel: a GPSS kernel as defined in flexible_function.py
        :param data: object of type GPCData which the kernel works on
        :param depth: depth of the current node in the search tree (root is 0)
        :param parent: the parent GPCKernel object (for back-tracking)
        """
        self.kernel = gpssKernel
        self.data = data
        self.depth = depth
        self.parent = parent
        self.model = None
        self.isSparse = None

        if not isinstance(self.kernel, ff.NoneKernel):
            self.kernel.initialise_params(data_shape=self.data.getDataShape())


    def __repr__(self):
        kernel_str = self.kernel.pretty_print()
        if isinstance(kernel_str, Exception):
            kernel_str = "NoneKernel"
        return 'GPCKernel: depth = %d, NLML = %f, CV error = %.4f\n' % \
               (self.depth, self.getNLML(), self.error()) + '  ' + \
               kernel_str


    def equals(self, other, strict=False, canonical=True):
        """
        Check if this kernel is equivalent to another kernel.

        :param other: kernel of type `GPCKernel` which is to be compared
        :param strict: check the equality of hyperparameters; default to False
        :param canonical: convert the kernel to canonical form before comparison;
        default to True
        :returns: True if the two kernels are equivalent, False otherwise
        """
        return isKernelEqual(self.kernel, other.kernel, compare_params=strict, use_canonical=canonical)


    def add(self, other):
        """
        Create a sum kernel by adding another kernel to the current kernel.

        :param other: `GPCKernel` object to be added to the current kernel
        :returns: `GPCKernel` object representing the sum kernel
        """
        k = self.kernel + other.kernel
        ret = GPCKernel(k, self.data)

        X, Y = self.data.X, self.data.Y
        ker = gpss2gpy(k, data=self.data)
        ret.isSparse = self.isSparse
        if ret.isSparse:
            # TODO: inherit number of inducing points from parent
            inducing = 10
            i = np.random.permutation(X.shape[0])[:inducing]
            Z = X[i].copy()
            lik = GPy.likelihoods.Bernoulli()
            ret.model = GPy.core.SVGP(X, Y, Z, ker, lik)
        else:
            ret.model = GPy.models.GPClassification(X, Y, kernel=ker)

        return ret


    def expand(self, base_kernels='SE'):
        """
        Expand this kernel using grammar defined in grammar.py.
        :returns: list of GPCKernel resulting from the expansion
        """
        ndim = self.data.getDim()
        g = grammar.MultiDGrammar(ndim, base_kernels=base_kernels, rules=None)
        kernels = grammar.expand(self.kernel, g)
        kernels = [k.canonical() for k in kernels]
        kernels = ff.remove_duplicates(kernels)
        for k in kernels:
            k.initialise_params(data_shape=self.data.getDataShape())
        kernels = [k.simplified() for k in kernels]
        kernels = ff.remove_duplicates(kernels)
        kernels = [k for k in kernels if not isinstance(k, ff.NoneKernel)]
        kernels = [GPCKernel(k, self.data, depth=self.depth+1, parent=self) for k in kernels]
        return kernels


    def reset(self):
        """
        Reset the kernel hyperparameters to random values.
        """
        self.kernel = removeKernelParams(self.kernel)
        if not isinstance(self.kernel, ff.NoneKernel):
            self.kernel.initialise_params(data_shape=self.data.getDataShape())
        self.isSparse = None
        self.errorRate = None


    def train(self, mode='auto', n_folds=5):
        """
        Train a GP classification model using k-fold cross-validation and random
        restart

        :param mode: 'full' for full GP, 'svgp' for scalable variational GP,
        'auto' for automatically selected model (default)
        :type mode: str
        :param n_folds: number of folds. If None, use current model
        hyperparameters as initial values; if `n_folds` is an integer, always
        randomise the initialisation before training, even if `n_folds` is 1
        :type n_folds: None or int
        """
        mode = mode.lower()
        assert mode in set(['full', 'svgp', 'auto']), "mode must be 'full', 'svgp' or 'auto'"
        assert n_folds is None or (isinstance(n_folds, int) and n_folds > 0), "n_folds must be None or positive integer"

        # Configure GP mode
        # TODO: threshold of data quantity for using SVGP instead of full inference
        if mode == 'auto':
            mode = 'full' if (self.data.getDim() * self.data.getNum() <= 4e3) else 'svgp'
        self.isSparse = mode == 'svgp'

        # Configure k-fold cross-validation and randomisation
        randomise = n_folds is not None
        if n_folds is None: n_folds = 1

        # Split dataset into training and validation sets
        X, Y, XT, YT = self.data.kFoldSplits(k=n_folds)

        # Train the appropriate GP model
        if self.isSparse:
            results = [self.trainSVGP(X[i], Y[i], XT=XT[i], YT=YT[i], randomise=randomise) for i in xrange(n_folds)]
        else:
            results = [self.trainFull(X[i], Y[i], XT=XT[i], YT=YT[i], randomise=randomise) for i in xrange(n_folds)]

        # Use kernel with median cross-validated error rate in a k-fold test
        # Record mean cross-validated error rate as overall performance
        if len(results) > 0:
            med = len(results) / 2
            sorted(results, key=lambda x: x['error'])
            self.model = results[med]['model']
            self.kernel = gpy2gpss(self.model.kern)
            self.errorRate = np.mean([x['error'] for x in results])
        else:
            print "Warning: none of the %d optimisation attempts were successful." % n_folds


    def trainFull(self, X, Y, XT=None, YT=None, randomise=False):
        """
        Train a full GP classification model using all data points, and compute
        cross-validated error rate on validation set
        Note that this method does NOT mutate this `GPCKernel` object. Instead
        it returns a trained `GPy.Model` object. To train the model AND update
        e.g. `self.model`, `self.kernel` fields, you have to call
        `GPCKernel.train()` method.

        :param X: training data points
        :param Y: training targets
        :param XT: validation data points, same as training if None
        :param YT: validation targets, same as training if None
        :param randomise: whether to randomise initial hyperparameters before
        optimising the model (default to False)
        :type randomise: bool
        :returns: trained `GPy.models.GPClassification` object and
        cross-validated error rate
        """
        if XT is None or YT is None:
            XT, YT = X, Y

        k = self.kernel
        if randomise:
            k = removeKernelParams(k)
            k.initialise_params(data_shape=self.data.getDataShape())

        m = GPy.models.GPClassification(X, Y, kernel=gpss2gpy(k, data=self.data))
        m.optimize()
        cverror = computeError(m, XT, YT)

        return {
            'model': m,
            'error': cverror
        }


    def trainSVGP(self, X, Y, XT=None, YT=None, randomise=False, inducing=10):
        """
        Train a sparse GP classification model using scalable variational GP,
        and compute cross-validated error rate on validation set
        Note that this method does NOT mutate this `GPCKernel` object. Instead
        it returns a trained `GPy.Model` object. To train the model AND update
        e.g. `self.model`, `self.kernel` fields, you have to call
        `GPCKernel.train()` method.

        :param X: training data points
        :param Y: training targets
        :param XT: validation data points, same as training if None
        :param YT: validation targets, same as training if None
        :param randomise: whether to randomise initial hyperparameters before
        optimising the model (default to False)
        :type randomise: bool
        :param inducing: number of inducing points to use
        :type inducing: int
        :returns: trained `GPy.core.SVGP` object and cross-validated error rate
        """
        if XT is None or YT is None:
            XT, YT = X, Y

        k = self.kernel
        if randomise:
            k = removeKernelParams(k)
            k.initialise_params(data_shape=self.data.getDataShape())

        i = np.random.permutation(X.shape[0])[:inducing]
        Z = X[i].copy()
        ker = gpss2gpy(k, data=self.data)
        lik = GPy.likelihoods.Bernoulli()
        m = GPy.core.SVGP(X, Y, Z, ker, lik)
        m.optimize()
        cverror = computeError(m, XT, YT)

        return {
            'model': m,
            'error': cverror
        }


    def toSummands(self):
        """
        Convert to sum of products

        :returns: list of GPCKernel objects which are additive components of
        the current kernel
        """
        k = self.kernel.additive_form()
        if isinstance(k, ff.SumKernel):
            summands = [GPCKernel(o, self.data) for o in k.operands]
        else:
            summands = [GPCKernel(k, self.data)]

        X, Y = self.data.X, self.data.Y
        for s in summands:
            s.isSparse = self.isSparse

            ker = gpss2gpy(s.kernel, data=self.data)
            if s.isSparse:
                # TODO: inherit number of inducing points from parent
                inducing = 10
                i = np.random.permutation(X.shape[0])[:inducing]
                Z = X[i].copy()
                lik = GPy.likelihoods.Bernoulli()
                s.model = GPy.core.SVGP(X, Y, Z, ker, lik)
            else:
                s.model = GPy.models.GPClassification(X, Y, kernel=ker)

        return summands


    def draw(self, filename, active_dims_only=False, draw_posterior=True):
        """
        Plot the model and data points

        :param filename: the output file (path and) name, without extension
        :param active_dims_only: True if want to present only the active
        dimensions (defaults to False)
        :param draw_posterior: True if want to draw the posterior contour
        (defaults to True)
        """
        if active_dims_only:
            plot = GPCPlot.create(self.model, xlabels=self.data.XLabel, usetex=True,
                active_dims=self.getActiveDims())
        else:
            plot = GPCPlot.create(self.model, xlabels=self.data.XLabel, usetex=True)

        plot.draw(draw_posterior=draw_posterior)
        plot.save(filename)


    def misclassifiedPoints(self, X=None, Y=None):
        """
        Find testing data points which are misclassified by the current model.

        :param X: testing data points, defaults to the entire current dataset
        :param Y: testing targets, defaults to the entire current dataset
        :returns: list of misclassified training points
        """
        model = self.model
        if X is None or Y is None: X, Y = self.data.X, self.data.Y
        return misclassifiedPoints(model, X, Y)


    def getDepth(self):
        """
        :returns: depth of this kernel in the search tree
        """
        return self.depth


    def getNLML(self):
        """
        :returns: negative log marginal likelihood
        """
        return float("inf") if self.model is None else -self.model.log_likelihood()


    def getActiveDims(self):
        """
        Active dimensions that the current kernel is working on.
        :returns: list of active dimensions
        """
        if self.model is not None:
            return list(self.model.kern.active_dims)
        else:
            return list(gpss2gpy(self.kernel, data=self.data).active_dims)


    def getGPyKernel(self):
        """
        Convert this GPCKernel to GPy kernel.
        :returns: an object of type GPy.kern.Kern
        """
        return gpss2gpy(self.kernel, data=self.data)


    def error(self):
        """
        Cached training error rate. This is usually the average k-fold
        cross-validated error rate. If no error rate is cached, this method will
        compute error rate over the entire training set (i.e. without
        cross-validation).
        """
        if isinstance(self.kernel, ff.ConstKernel):
            # TODO: this is ugly
            d = self.data
            return min(d.getClass(0).shape[0], d.getClass(1).shape[0]) / float(d.getNum())

        if not hasattr(self, 'errorRate') or self.errorRate is None:
            self.errorRate = computeError(self.model, self.data.X, self.data.Y)
        return self.errorRate


    def monotonicity(self, margin=0.15):
        """
        Test if a 1-D kernel has monotonic posterior mean.

        :param margin: fraction of the input range to be discarded on each extreme.
        We only run tests on the middle part of the input range, as boundary values
        can have non-monotonic latent function mean values
        :returns: 1 if increasing, -1 if decreasing, 0 if non-monotonic
        """
        assert len(self.getActiveDims()) == 1, 'Kernel must be one-dimensional'

        if margin < 0: margin = 0
        if margin > 0.5: margin = 0.5

        dim = self.getActiveDims()[0]
        x = self.data.X[:,dim]
        xmin, xmax = x.min(), x.max()
        xlo = xmin + margin * (xmax - xmin)
        xhi = xmax - margin * (xmax - xmin)
        X = self.data.X[(x >= xlo) & (x <= xhi)]
        X = X[X[:,dim].argsort()]
        dmu_dx, _ = self.model.predictive_gradients(X)
        dmu_dx = dmu_dx[:,dim,0].reshape((-1,1))

        if np.all(dmu_dx > 0):
            return 1
        elif np.all(dmu_dx < 0):
            return -1
        else:
            return 0


    def period(self):
        """
        Period of a 1-D periodic kernel.

        :returns: period of a periodic kernel, or 0 if not periodic
        """
        assert len(self.getActiveDims()) == 1, 'Kernel must be one-dimensional'
        if isinstance(self.kernel, ff.PeriodicKernel):
            return self.kernel.period
        else:
            return 0.0


    def shortInterp(self):
        """
        Interpretation of current kernel:
        SE       - smooth
        Periodic - periodic
        Const    - constant
        Sum      - additive
        Prod     - interaction
        None     - null
        """
        k = self.kernel
        if isinstance(k, ff.SqExpKernel):
            return "smooth"
        elif isinstance(k, ff.PeriodicKernel):
            return "periodic"
        elif isinstance(k, ff.ConstKernel):
            return "constant"
        elif isinstance(k, ff.SumKernel):
            return "additive"
        elif isinstance(k, ff.ProductKernel):
            return "interaction"
        elif isinstance(k, ff.NoneKernel):
            return "null"
        else:
            raise NotImplementedError("Unrecognised kernel type.")


##############################################
#                                            #
#             Helper Functions               #
#                                            #
##############################################

def gpss2gpy(kernel, data=None):
    """
    Convert a GPSS kernel to a GPy kernel recursively, applying constraints to
    parameters when appropriate.

    Support only:
    1) 1-D squared exponential kernels
    2) 1-D periodic kernels
    3) constant kernels (called `bias` in GPy)
    4) sum kernels
    5) product kernels

    :param kernel: GPSS kernel as defined in `flexible_function.py`
    :param data: `GPCData` object. If None (default), do not apply constraints
    :returns: object of type GPy.kern.Kern
    """
    assert isinstance(kernel, ff.Kernel), "kernel must be of type flexible_function.Kernel"

    # Hard-coded constraint on sf
    sf2min, sf2max = 0.1 ** 2, 20 ** 2

    if isinstance(kernel, ff.SqExpKernel):
        sf2 = kernel.sf ** 2
        ls = kernel.lengthscale
        dim = kernel.dimension
        gpyker = GPy.kern.RBF(1, variance=sf2, lengthscale=ls, active_dims=np.array([dim]))
        gpyker['variance'].constrain_bounded(sf2min, sf2max, warning=False)
        if data:
            gpyker['lengthscale'].constrain_bounded(data.minSeparation(dims=dim),
                data.inputRange(dims=dim) * 2, warning=False)
        return gpyker

    elif isinstance(kernel, ff.PeriodicKernel):
        sf2 = kernel.sf ** 2
        per = kernel.period
        ls = kernel.lengthscale
        dim = kernel.dimension
        gpyker = GPy.kern.StdPeriodic(1, variance=sf2, period=per, lengthscale=ls, active_dims=np.array([dim]))
        gpyker['variance'].constrain_bounded(sf2min, sf2max, warning=False)
        if data:
            gpyker['lengthscale'].constrain_bounded(data.minSeparation(dims=dim),
                data.inputRange(dims=dim) * 2, warning=False)
            gpyker['period'].constrain_bounded(data.minSeparation(dims=dim) * 2,
                data.inputRange(dims=dim), warning=False)
        return gpyker

    elif isinstance(kernel, ff.ConstKernel):
        assert isinstance(data, GPCData), 'Must specify data field for ConstKernel'
        sf2 = kernel.sf ** 2
        ndim = data.getDim()
        gpyker = GPy.kern.Bias(ndim, variance=sf2, active_dims=np.array(range(ndim)))
        return gpyker

    elif isinstance(kernel, ff.SumKernel):
        return GPy.kern.Add([gpss2gpy(o, data=data) for o in kernel.operands])

    elif isinstance(kernel, ff.ProductKernel):
        return GPy.kern.Prod([gpss2gpy(o, data=data) for o in kernel.operands])

    else:
        raise NotImplementedError("Cannot translate kernel of type " + type(kernel).__name__)


def gpy2gpss(kernel):
    """
    Convert a GPy kernel to a GPSS kernel recursively.

    Support only:
    1) 1-D squared exponential kernels
    2) 1-D periodic kernels
    3) constant kernels (called `bias` in GPy)
    4) sum kernels
    5) product kernels

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
        ls = kernel.lengthscale[0]
        per = kernel.period[0]
        dim = kernel.active_dims[0]
        return ff.PeriodicKernel(dimension=dim, lengthscale=ls, period=per, sf=sf)

    elif isinstance(kernel, GPy.kern.Bias):
        sf = np.sqrt(kernel.variance)[0]
        return ff.ConstKernel(sf=sf)

    elif isinstance(kernel, GPy.kern.Add):
        return ff.SumKernel(map(gpy2gpss, kernel.parts))

    elif isinstance(kernel, GPy.kern.Prod):
        return ff.ProductKernel(map(gpy2gpss, kernel.parts))

    else:
        raise NotImplementedError("Cannot translate kernel of type " + type(kernel).__name__)


def removeKernelParams(kernel):
    """
    Remove hyperparameters of a GPSS kernel and reset them to None.

    :returns: a GPSS kernel without parameter initialisation
    """
    assert isinstance(kernel, ff.Kernel), "kernel must be of type flexible_function.Kernel"

    if isinstance(kernel, ff.SqExpKernel):
        return ff.SqExpKernel(dimension=kernel.dimension)

    elif isinstance(kernel, ff.PeriodicKernel):
        return ff.PeriodicKernel(dimension=kernel.dimension)

    elif isinstance(kernel, ff.ConstKernel):
        return ff.ConstKernel()

    elif isinstance(kernel, ff.SumKernel):
        return ff.SumKernel(map(removeKernelParams, kernel.operands))

    elif isinstance(kernel, ff.ProductKernel):
        return ff.ProductKernel(map(removeKernelParams, kernel.operands))

    elif isinstance(kernel, ff.NoneKernel):
        return kernel

    else:
        raise NotImplementedError("Unrecognised kernel type " + type(kernel).__name__)


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


def misclassifiedPoints(model, XT, YT):
    """
    Samples misclassified by given GP classifier.

    :param model: GPy model
    :param XT: testing data points
    :param YT: testing set targets
    :returns: array of misclassified points
    """
    if model is None:
        return {'X': XT, 'Y': YT}

    Phi, _ = model.predict(XT)                     # Predicted Y, range [0, 1]
    OK = (((Phi - 0.5) * (YT - 0.5)) < 0).flatten()     # < 0 if misclassified
    return {'X': XT[OK], 'Y': YT[OK]}


def computeError(model, XT, YT):
    """
    Compute training error of given GP classifier.

    :param model: GPy model
    :param XT: testing data points
    :param YT: testing set targets
    :returns: error rate in range [0, 1]
    """
    return misclassifiedPoints(model, XT, YT)['X'].shape[0] / float(XT.shape[0])

