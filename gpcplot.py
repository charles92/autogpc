# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from GPy.plotting.matplot_dep.base_plots import x_frame1D

# Figure DPI
default_dpi = 1200
# Number of samples drawn from the posterior GP, which are then plotted
default_res = 500

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
        print 'TODO: draw 1D'
        m = self.model
        fig, ax = plt.subplots()

        # Data range
        xnew, xmin, xmax = x_frame1D(m.X, resolution=default_res)

        # Data points
        ax.plot(m.X, m.Y, label='Training data', linestyle='',
            marker='x', mfc='blue', mew=1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin=-0.2, ymax=1.2)
        ax.set_yticks([0, 0.5, 1])

        # Latent function
        mu, var = m._raw_predict(xnew)
        stdev = np.sqrt(var)
        lower = m.likelihood.gp_link.transf(mu - 2 * stdev)
        upper = m.likelihood.gp_link.transf(mu + 2 * stdev)
        mu = m.likelihood.gp_link.transf(mu)
        plotGP(xnew, mu, lower=lower, upper=upper, ax=ax)

        self.fig = fig


class GPCPlot2D(GPCPlot):
    """
    Gaussian process classification plot: 2-dimensional input

    """

    def __init__(self, model, dpi=default_dpi):
        GPCPlot.__init__(self, model, dpi)

    def draw(self):
        print 'TODO: 2D'


class GPCPlot3D(GPCPlot):
    """
    Gaussian process classification plot: 3-dimensional input

    """

    def __init__(self, model, dpi=default_dpi):
        GPCPlot.__init__(self, model, dpi)

    def draw(self):
        print 'TODO: 3D'


class GPCPlotHD(GPCPlot):
    """
    Gaussian process classification plot: high (> 3) dimensional input

    """

    def __init__(self, model, dpi=default_dpi):
        GPCPlot.__init__(self, model, dpi)

    def draw(self):
        print 'TODO: HD'

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
