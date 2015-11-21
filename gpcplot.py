# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab

# Number of samples drawn from the posterior GP, which are then plotted
default_res = 256

# Number of contour levels
default_lvl = 7


class GPCPlot(object):
    """
    Gaussian process classification plot

    """

    def create(model):
        input_dim = model.input_dim
        if input_dim == 1:
            return GPCPlot1D(model)
        elif input_dim == 2:
            return GPCPlot2D(model)
        elif input_dim == 3:
            return GPCPlot3D(model)
        elif input_dim >= 4:
            return GPCPlotHD(model)
        else:
            raise ValueError('The model must have >= 1 input dimension.')
    create = staticmethod(create)

    def __init__(self, model):
        self.model = model

    def draw(self):
        raise NotImplementedError

    def save(self, fname):
        self.fig.savefig(fname + '.eps')
        plt.close(self.fig)
        print 'DEBUG: GPCPlot.save(): fname={}'.format(fname + '.eps')


class GPCPlot1D(GPCPlot):
    """
    Gaussian process classification plot: 1-dimensional input

    """

    def __init__(self, model):
        GPCPlot.__init__(self, model)

    def draw(self):
        m = self.model
        fig, ax = plt.subplots()
        plots = {}

        # Data range
        xmin, xmax, _, xgrd = getFrame(m.X)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin=-0.2, ymax=1.2)
        ax.set_yticks([0, 0.5, 1])

        # Data points
        plots['data'] = ax.plot(m.X, m.Y, label='Training data', linestyle='',
            marker='x', mfc='blue', mew=1)

        # Latent function
        mu, var = m._raw_predict(xgrd)
        stdev = np.sqrt(var)
        lower = m.likelihood.gp_link.transf(mu - 2 * stdev)
        upper = m.likelihood.gp_link.transf(mu + 2 * stdev)
        mu = m.likelihood.gp_link.transf(mu)
        plots['link'] = plotGP(xgrd, mu, lower=lower, upper=upper, ax=ax)

        self.fig = fig
        return plots


class GPCPlot2D(GPCPlot):
    """
    Gaussian process classification plot: 2-dimensional input

    """

    def __init__(self, model):
        GPCPlot.__init__(self, model)

    def draw(self):
        m = self.model
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        plots = {}

        # Data range
        xmin, xmax, xrng, xgrd = getFrame(m.X)
        ax0.set_xlim(xmin[0], xmax[0])
        ax0.set_ylim(xmin[1], xmax[1])

        # Data points
        plots['data1'] = ax0.scatter(m.X[:,0], m.X[:,1], c=m.Y, marker='o',
            edgecolors='none', alpha=0.2, vmin=-0.2, vmax=1.2, cmap=plt.cm.jet)
        plots['data2'] = ax1.scatter(m.X[:,0], m.X[:,1], c=m.Y, marker='o',
            edgecolors='none', alpha=0.2, vmin=-0.2, vmax=1.2, cmap=plt.cm.jet)

        # Latent function
        mu, var = m._raw_predict(xgrd)

        # Latent function - mean
        mu = mu.reshape(default_res, default_res).T
        cs = ax0.contour(xrng[:,0], xrng[:,1], mu, default_lvl,
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
        cs = ax1.contour(xrng[:,0], xrng[:,1], sd, default_lvl // 2,
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

    def __init__(self, model):
        GPCPlot.__init__(self, model)

    def draw(self):
        m = self.model
        fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
        plots = {}

        xmin, xmax, xrng, xgrd = getFrame(m.X, res=32)

        # Data points
        pts3d = mlab.points3d(m.X[:,0], m.X[:,1], m.X[:,2], m.Y[:,0],
            extent=np.vstack((xmin, xmax)).T.flatten(), figure=fig,
            mode='sphere', vmin=-0.2, vmax=1.2, colormap='jet',
            scale_mode='none', scale_factor=0.05)
        mlab.outline(pts3d)
        mlab.axes(pts3d)
        plots['data'] = pts3d

        # Contour surfaces of GP mean
        mu, _ = m._raw_predict(xgrd)
        xx, yy, zz = np.meshgrid(*tuple(xrng[:,i] for i in range(3)), indexing='ij')
        mu = mu.reshape(xx.shape)
        plots['gpmu'] = mlab.contour3d(xx, yy, zz, mu, figure=fig, colormap='jet',
            contours=[-1, 0, 1], opacity = 0.25, vmin=-1.5, vmax=1.5)

        self.fig = fig
        return plots

    def save(self, fname):
        mlab.view(azimuth=45, elevation=60, distance='auto', focalpoint='auto',
            figure=self.fig)
        mlab.savefig(fname + '-1.png', figure=self.fig)
        print 'DEBUG: GPCPlot3D.save(): fname={}'.format(fname + '-1.png')

        mlab.view(azimuth=225, elevation=120, distance='auto', focalpoint='auto',
            figure=self.fig)
        mlab.savefig(fname + '-2.png', figure=self.fig)
        print 'DEBUG: GPCPlot3D.save(): fname={}'.format(fname + '-1.png')

        mlab.close(scene=self.fig)


class GPCPlotHD(GPCPlot):
    """
    Gaussian process classification plot: high (> 3) dimensional input

    """

    def __init__(self, model):
        GPCPlot.__init__(self, model)

    def draw(self):
        print 'TODO: HD'


def getFrame(X, res=default_res):
    """
    Calculate the optimal frame for plotting data points.

    Arguments:
    X -- Data points as a N-by-D matrix, where N is the number of data points
    and D is the number of dimensions in each data point

    Keyword arguments:
    res -- Number of subsamples in each dimension of X (default 256)

    Returns:
    xmin, xmax, xrng, xgrd
    xmin -- 1-by-D matrix, the lower limit of plotting frame in each dimension
    xmax -- 1-by-D matrix, the upper limit of plotting frame in each dimension
    xrng -- res-by-D matrix, evenly spaced sampling points in each dimension
    xgrd -- (res^D)-by-D matrix, the meshgrid stacked in the same way as X
    """

    xmin, xmax = X.min(axis=0), X.max(axis=0)
    margin = 0.2 * (xmax - xmin)
    xmin, xmax = xmin - margin, xmax + margin

    xdim = X.shape[1]
    xrng = np.vstack(tuple(np.linspace(x1, x2, num=res) \
        for x1,x2 in zip(xmin,xmax))).T

    xgrd = np.meshgrid(*tuple(xrng[:,i] for i in range(xdim)), indexing='ij')
    xgrd = np.hstack(tuple(xgrd[i].reshape(-1,1) for i in range(xdim)))

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
