# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from GPy.plotting.matplot_dep.base_plots import x_frame1D, x_frame2D

# Figure DPI
default_dpi = 300

# Number of samples drawn from the posterior GP, which are then plotted
default_res = 256

# Number of contour levels
default_lvl = 7


class GPCPlot(object):
    """
    Gaussian process classification plot

    """

    def create(model, dpi=default_dpi):
        input_dim = model.input_dim
        if input_dim == 1:
            return GPCPlot1D(model, dpi)
        elif input_dim == 2:
            return GPCPlot2D(model, dpi)
        elif input_dim == 3:
            return GPCPlot3D(model, dpi)
        elif input_dim >= 4:
            return GPCPlotHD(model, dpi)
        else:
            raise ValueError('The model must have >= 1 input dimension.')
    create = staticmethod(create)

    def __init__(self, model, dpi=default_dpi):
        self.model = model
        self.dpi = dpi

    def draw(self):
        raise NotImplementedError

    def save(self, fname):
        self.fig.savefig(fname, dpi=self.dpi)
        print 'DEBUG: GPCPlot.save(): fname={}, dpi={}'.format(fname, self.dpi)


class GPCPlot1D(GPCPlot):
    """
    Gaussian process classification plot: 1-dimensional input

    """

    def __init__(self, model, dpi=default_dpi):
        GPCPlot.__init__(self, model, dpi)

    def draw(self):
        m = self.model
        fig, ax = plt.subplots()
        plots = {}

        # Data range
        xnew, xmin, xmax = x_frame1D(m.X, resolution=default_res)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin=-0.2, ymax=1.2)
        ax.set_yticks([0, 0.5, 1])

        # Data points
        plots['data'] = ax.plot(m.X, m.Y, label='Training data', linestyle='',
            marker='x', mfc='blue', mew=1)

        # Latent function
        mu, var = m._raw_predict(xnew)
        stdev = np.sqrt(var)
        lower = m.likelihood.gp_link.transf(mu - 2 * stdev)
        upper = m.likelihood.gp_link.transf(mu + 2 * stdev)
        mu = m.likelihood.gp_link.transf(mu)
        plots['link'] = plotGP(xnew, mu, lower=lower, upper=upper, ax=ax)

        self.fig = fig
        return plots


class GPCPlot2D(GPCPlot):
    """
    Gaussian process classification plot: 2-dimensional input

    """

    def __init__(self, model, dpi=default_dpi):
        GPCPlot.__init__(self, model, dpi)

    def draw(self):
        m = self.model
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        plots = {}

        # Data range
        xnew, _, _, xmin, xmax = x_frame2D(m.X, resolution=default_res)
        ax0.set_xlim(xmin[0], xmax[0])
        ax0.set_ylim(xmin[1], xmax[1])

        # Data points
        plots['data1'] = ax0.scatter(m.X[:,0], m.X[:,1], c=m.Y, marker='o',
            edgecolors='none', alpha=0.2, vmin=-0.2, vmax=1.2, cmap=plt.cm.jet)
        plots['data2'] = ax1.scatter(m.X[:,0], m.X[:,1], c=m.Y, marker='o',
            edgecolors='none', alpha=0.2, vmin=-0.2, vmax=1.2, cmap=plt.cm.jet)

        # Latent function
        x0range = np.linspace(xmin[0], xmax[0], default_res)
        x1range = np.linspace(xmin[1], xmax[1], default_res)
        mu, var = m._raw_predict(xnew)

        # Latent function - mean
        mu = mu.reshape(default_res, default_res).T
        cs = ax0.contour(x0range, x1range, mu, default_lvl,
            vmin=mu.min(), vmax=mu.max(), cmap=plt.cm.jet)
        # Make zero contour thicker
        if np.all(cs.levels != 0):
            cs.levels = np.hstack((cs.levels, 0))
        zcind = np.where(cs.levels == 0)[0].flatten()
        plt.setp(cs.collections[zcind], linewidth=2)
        # Add contour labels
        ax0.clabel(cs, fontsize=8)
        plots['gpmu'] = cs

        # Latent function - standard deviation
        var = var.reshape(default_res, default_res).T
        sd = np.sqrt(var)
        cs = ax1.contour(x0range, x1range, sd, default_lvl // 2,
            vmin=sd.min(), vmax=sd.max(), cmap=plt.cm.OrRd)
        # Add contour labels
        ax1.clabel(cs, fontsize=8)
        plots['gpsd'] = cs

        self.fig = fig
        return plots


class GPCPlot3D(GPCPlot):
    """
    Gaussian process classification plot: 3-dimensional input

    """

    def __init__(self, model, dpi=default_dpi):
        GPCPlot.__init__(self, model, dpi)

    def draw(self):
        print 'TODO: 3D'
        m = self.model
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')
        plots = {}

        # Data points
        plots['data'] = ax.scatter(m.X[:,0], m.X[:,1], m.X[:,2], c=m.Y,
            marker='o', edgecolors='none', depthshade=True,
            vmin=-0.2, vmax=1.2, cmap=plt.cm.jet)

        # Zero contour surface
        xmin, xmax, xrng, xgrd = getFrame(m.X, resolution=64)
        mu, _ = m._raw_predict(xgrd)
        zc = findZeros3D(xrng, mu.reshape((xrng.shape[0],) * 3))
        plots['gpzerocontour'] = ax.plot_trisurf(zc[:,0], zc[:,1], zc[:,2],
            color='green', alpha=0.2, linewidth=0)

        self.fig = fig
        return plots


class GPCPlotHD(GPCPlot):
    """
    Gaussian process classification plot: high (> 3) dimensional input

    """

    def __init__(self, model, dpi=default_dpi):
        GPCPlot.__init__(self, model, dpi)

    def draw(self):
        print 'TODO: HD'


def getFrame(X, resolution=default_res):
    """
    Calculate the optimal frame for plotting data points whose x coordinates
    (x0, x1, ...) are given in matrix X.

    """
    xmin, xmax = X.min(0), X.max(0)
    margin = 0.2 * (xmax - xmin)
    xmin, xmax = xmin - margin, xmax + margin

    xdim = X.shape[1]
    xrng = np.vstack(tuple(np.linspace(x1, x2, num=resolution) \
        for x1,x2 in zip(xmin,xmax))).T

    grids = np.meshgrid(*tuple(xrng[:,i] for i in range(xdim)), indexing='ij')
    xgrd = np.hstack(tuple(grids[i].reshape(-1,1) for i in range(xdim)))

    return xmin, xmax, xrng, xgrd


def plotGP(x, mu, lower=None, upper=None, ax=None,
    meancolor='blue', edgecolor='black', fillcolor='#DDDDDD',
    meanwidth=2, edgewidth=0.25):
    """
    Make a generic 1-D GP plot on certain axes, with optional error band

    """
    if ax is None:
        _, ax = plt.subplots()

    plots = {}

    # Mean
    plots['mean'] = ax.plot(x, mu, color=meancolor, linewidth=meanwidth)

    if lower is not None and upper is not None:
        # Lower and upper edges
        plots['lower'] = ax.plot(x, lower, color=edgecolor, linewidth=edgewidth)
        plots['upper'] = ax.plot(x, upper, color=edgecolor, linewidth=edgewidth)

        # Fill between edges
        plots['fill'] = ax.fill(np.vstack((x,x[::-1])),
            np.vstack((upper,lower[::-1])),
            color=fillcolor)

    return plots


def findZeros3D(X, Y):
    """
    Find zero crossings of function Y=f(X), where X is 3-dimensional.
    X is a tuple of column vectors, corresponding to the sampling points in each
    axis.

    """
    if isinstance(X, tuple) and len(X) == 3:
        X0 = X[0].flatten()
        X1 = X[1].flatten()
        X2 = X[2].flatten()
    elif isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == 3:
        X0 = X[:,0]
        X1 = X[:,1]
        X2 = X[:,2]
    else:
        raise ValueError('Invalid input arguments.')

    assert (X0.size, X1.size, X2.size) == Y.shape, \
    'The shape of output Y does not match that of the input X.'

    Z = np.empty((0, 3))

    for i0,x0 in enumerate(X0):
        for i1,x1 in enumerate(X1):
            for x2 in findZeros1D(X2, Y[i0,i1,:]):
                Z = np.vstack((Z, [x0, x1, x2]))

    return Z

def findZeros1D(X, Y):
    """
    Find zero crossings of univariate function Y=f(X).
    X, Y are vectors of the same length.

    """
    X0 = X.flatten()
    Y0 = Y.flatten()
    assert X0.size == Y0.size, 'X, Y must be vectors of the same length.'

    sgn = np.sign(Y0)
    sgn[sgn == 0] = -1
    zcx = X0[np.where(np.diff(sgn))]

    return zcx
