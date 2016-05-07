# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

# Set ETS_TOOLKIT to qt4 instead of wxPython. The latter crashes on Mac. Don't
# know about stability on other platforms.
# Testing platform: Mac OS X 10.11.1
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import moviepy.editor as mpy

# Number of samples drawn from the posterior GP, which are then plotted
default_res = 256

# Posterior levels at which contours are drawn
default_contours = [0.05, 0.25, 0.5, 0.75, 0.95]
default_sd_contours = [0.05, 0.2, 0.5]


class GPCPlot(object):
    """
    Gaussian process classification plot
    """

    @staticmethod
    def create(model, active_dims=None, xlabels=None, usetex=False):
        if active_dims is None or len(active_dims) == 0:
            active_dims = range(model.input_dim)
        else:
            assert max(active_dims) < model.input_dim, 'Error: active dimension out of bound.'

        input_dim = len(active_dims)
        if input_dim == 1:
            return GPCPlot1D(model, active_dims, xlabels=xlabels, usetex=usetex)
        elif input_dim == 2:
            return GPCPlot2D(model, active_dims, xlabels=xlabels, usetex=usetex)
        elif input_dim == 3:
            return GPCPlot3D(model, active_dims, xlabels=xlabels, usetex=usetex)
        elif input_dim >= 4:
            return GPCPlotHD(model, active_dims, xlabels=xlabels, usetex=usetex)
        else:
            raise ValueError('The model must have >= 1 input dimension.')

    def __init__(self, model, active_dims, xlabels, usetex):
        """
        This constructor should not be called directly. All instantiation of
        GPCPlot objects should be done via the factory method GPCPlot.create().
        """
        assert model is not None, 'GP model must not be None.'
        assert xlabels is not None, 'Labels for X axes must not be None.'
        self.model = model
        self.active_dims = active_dims
        self.xlabels = xlabels
        self.usetex = usetex

    def draw(self, draw_posterior=True):
        raise NotImplementedError

    def save(self, fname):
        self.fig.savefig(fname + '.eps')
        plt.close(self.fig)
        print 'DEBUG: GPCPlot.save(): fname={}'.format(fname + '.eps')


class GPCPlot1D(GPCPlot):
    """
    Gaussian process classification plot: 1-dimensional input
    """

    def __init__(self, model, active_dims, xlabels=None, usetex=False):
        assert len(active_dims) == 1, 'Error: GPCPlot1D only accepts 1 active dimension'
        if isinstance(xlabels, (list, tuple)) and len(xlabels) > active_dims[0]:
            xlabels = [xlabels[active_dims[0]]]
        else:
            xlabels = (r'$x$',)
            usetex = True
        GPCPlot.__init__(self, model, active_dims, xlabels, usetex)

    def draw(self, draw_posterior=True):
        m = self.model
        plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        plots = {}

        # Data range
        active_X = m.X[:,self.active_dims]
        xmin, xmax, _, xgrd = getFrame(active_X)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin=-0.2, ymax=1.2)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(r'$\phi(f)$')
        plt.rc('text', usetex=self.usetex)
        ax.set_xlabel(self.xlabels[0])
        plt.rc('text', usetex=True)

        # Data points
        plots['data'] = ax.scatter(active_X, m.Y, c=m.Y, marker='o',
            edgecolors='none', alpha=0.2, vmin=-0.2, vmax=1.2, cmap=plt.cm.jet)

        # Latent function with 95% confidence interval
        if draw_posterior:
            fullxgrd = np.zeros((xgrd.shape[0], m.input_dim))
            fullxgrd[:,self.active_dims] = xgrd
            mu, var = m._raw_predict(fullxgrd)
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

    def __init__(self, model, active_dims, xlabels=None, usetex=False):
        assert len(active_dims) == 2, 'Error: GPCPlot2D only accepts 2 active dimensions'
        if isinstance(xlabels, (list, tuple)) and len(xlabels) > max(active_dims):
            xlabels = [xlabels[d] for d in active_dims]
        else:
            xlabels = (r'$x_1$', r'$x_2$')
            usetex = True
        GPCPlot.__init__(self, model, active_dims, xlabels, usetex)

    def draw(self, draw_posterior=True):
        m = self.model
        plt.rc('text', usetex=True)
        plots = {}
        if draw_posterior:
            fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        else:
            fig, ax0 =  plt.subplots()

        # Data range
        active_X = m.X[:,self.active_dims]
        xmin, xmax, xrng, xgrd = getFrame(active_X)
        ax0.set_xlim(xmin[0], xmax[0])
        ax0.set_ylim(xmin[1], xmax[1])
        plt.rc('text', usetex=self.usetex)
        ax0.set_xlabel(self.xlabels[0])
        if draw_posterior: ax1.set_xlabel(self.xlabels[0])
        ax0.set_ylabel(self.xlabels[1])
        plt.rc('text', usetex=True)

        # Data points
        plots['data1'] = ax0.scatter(active_X[:,0], active_X[:,1], c=m.Y, marker='o',
            edgecolors='none', alpha=0.2, vmin=-0.2, vmax=1.2, cmap=plt.cm.jet)
        if draw_posterior:
            plots['data2'] = ax1.scatter(active_X[:,0], active_X[:,1], c=m.Y, marker='o',
                edgecolors='none', alpha=0.2, vmin=-0.2, vmax=1.2, cmap=plt.cm.jet)

        # Latent function
        if draw_posterior:
            fullxgrd = np.zeros((xgrd.shape[0], m.input_dim))
            fullxgrd[:,self.active_dims] = xgrd
            mu, var = m._raw_predict(fullxgrd)
            sd = np.sqrt(var)
            sd = m.likelihood.gp_link.transf(mu + 2 * sd) - m.likelihood.gp_link.transf(mu - 2 * sd)
            mu = m.likelihood.gp_link.transf(mu)

            # Latent function - mean
            mu = mu.reshape(default_res, default_res).T
            cs = ax0.contour(xrng[:,0], xrng[:,1], mu, default_contours,
                vmin=0, vmax=1, cmap=plt.cm.jet)
            # Make 0.5 contour thicker
            if np.any(cs.levels == 0.5):
                cind = np.where(cs.levels == 0.5)[0].flatten()
                plt.setp(cs.collections[cind], linewidth=2)
            # Add contour labels
            ax0.clabel(cs, fontsize=8)
            plots['gpmu'] = cs

            # Latent function - standard deviation
            sd = sd.reshape(default_res, default_res).T
            cs = ax1.contour(xrng[:,0], xrng[:,1], sd, default_sd_contours,
                vmin=-0.5, vmax=max(default_sd_contours), cmap=plt.cm.OrRd)
            # Add contour labels
            ax1.clabel(cs, fontsize=8)
            plots['gpsd'] = cs

        self.fig = fig
        return plots


class GPCPlot3D(GPCPlot):
    """
    Gaussian process classification plot: 3-dimensional input
    """

    def __init__(self, model, active_dims, xlabels=None, usetex=False):
        assert len(active_dims) == 3, 'Error: GPCPlot3D only accepts 3 active dimensions'
        if isinstance(xlabels, (list, tuple)) and len(xlabels) > max(active_dims):
            xlabels = [xlabels[d] for d in active_dims]
        else:
            xlabels = ('x1', 'x2', 'x3')
        if usetex:
            print 'Warning: usetex is not supported for 3-D plots. Using False instead.'
        GPCPlot.__init__(self, model, active_dims, xlabels, False)

    def draw(self, draw_posterior=True):
        m = self.model
        fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 601))
        plots = {}

        active_X = m.X[:,self.active_dims]
        xmin, xmax, xrng, xgrd = getFrame(active_X, res=32)

        # Data points
        pts3d = mlab.points3d(active_X[:,0], active_X[:,1], active_X[:,2], m.Y[:,0],
            extent=np.vstack((xmin, xmax)).T.flatten(), figure=fig,
            mode='sphere', vmin=-0.2, vmax=1.2, colormap='jet',
            scale_mode='none', scale_factor=0.05)
        mlab.outline(pts3d, color=(0.5, 0.5, 0.5))
        mlab.axes(pts3d, xlabel=self.xlabels[0], ylabel=self.xlabels[1],
            zlabel=self.xlabels[2])
        plots['data'] = pts3d

        # Contour surfaces of GP mean
        if draw_posterior:
            fullxgrd = np.zeros((xgrd.shape[0], m.input_dim))
            fullxgrd[:,self.active_dims] = xgrd
            mu, _ = m._raw_predict(fullxgrd)
            mu = m.likelihood.gp_link.transf(mu)
            xx, yy, zz = np.meshgrid(*tuple(xrng[:,i] for i in range(3)), indexing='ij')
            mu = mu.reshape(xx.shape)
            plots['gpmu'] = mlab.contour3d(xx, yy, zz, mu, figure=fig, colormap='jet',
                contours=[.1, .5, .9], opacity = 0.25, vmin=0, vmax=1)

        self.fig = fig
        return plots

    def save(self, fname, animate=False):
        # Animation
        def make_frame(t):
            t = t % 8
            if t < 3:
                az = (45 + 60 * t) % 360
                mlab.view(figure=self.fig, azimuth=az, elevation=60,
                    distance='auto', focalpoint='auto')
            elif t < 4:
                el = (60 + 60 * (t - 3)) % 180
                mlab.view(figure=self.fig, azimuth=225, elevation=el,
                    distance='auto', focalpoint='auto')
            elif t < 7:
                az = (225 - 60 * (t - 4)) % 360
                mlab.view(figure=self.fig, azimuth=az, elevation=120,
                    distance='auto', focalpoint='auto')
            else:
                el = (120 - 60 * (t - 7)) % 180
                mlab.view(figure=self.fig, azimuth=45, elevation=el,
                    distance='auto', focalpoint='auto')
            return mlab.screenshot(antialiased=True)
        if animate:
            anim = mpy.VideoClip(make_frame, duration=8)
            anim.write_videofile(fname + '.mp4', fps=24, audio=False, codec='libx264')
            print 'DEBUG: GPCPlot3D.save(): fname={}'.format(fname + '.mp4')
            anim.write_gif(fname + '.gif', fps=24)
            print 'DEBUG: GPCPlot3D.save(): fname={}'.format(fname + '.gif')

        # Static view 1
        mlab.view(figure=self.fig, azimuth=45, elevation=60,
            distance='auto', focalpoint='auto',)
        mlab.savefig(fname + '.png', figure=self.fig)
        print 'DEBUG: GPCPlot3D.save(): fname={}'.format(fname + '.png')

        # Static view 2
        mlab.view(figure=self.fig, azimuth=225, elevation=120,
            distance='auto', focalpoint='auto')
        mlab.savefig(fname + '-2.png', figure=self.fig)
        print 'DEBUG: GPCPlot3D.save(): fname={}'.format(fname + '-2.png')

        mlab.close(scene=self.fig)


class GPCPlotHD(GPCPlot):
    """
    Gaussian process classification plot: high (> 3) dimensional input
    """

    def __init__(self, model, active_dims, xlabels=None, usetex=False):
        assert len(active_dims) > 3, 'Error: GPCPlotHD only accepts >3 active dimensions'
        if isinstance(xlabels, (list, tuple)) and len(xlabels) > max(active_dims):
            xlabels = [xlabels[d] for d in active_dims]
        else:
            xlabels = tuple((r'$x_{' + str(i+1) + r'}$') for i in range(model.X.shape[1]))
            usetex = True
        GPCPlot.__init__(self, model, active_dims, xlabels, usetex)

    def draw(self, draw_posterior=True):
        m = self.model
        plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        plots = {}

        # Data range
        active_X = m.X[:,self.active_dims]
        xnum, xdim = active_X.shape
        xmin, xmax, _ = getFrame(active_X, grid=False)
        ax.axis(xmin=-0.1, xmax=xdim - 0.9, ymin=0, ymax=1)
        ax.axis('off')

        # Data axes
        axmargin = 0.1 # Fraction of the range of the data on each axis
        axmin = (1 - 5 * axmargin) / 7
        axmax = 1 - axmin
        dataaxes = []
        for i in range(xdim):
            dataaxis={}
            dataaxis['bg'] = ax.axvline(x=i, ymin=axmin, ymax=axmax,
                c='lightgray', lw=12)
            dataaxis['axis'] = ax.axvline(x=i, ymin=axmin, ymax=axmax,
                c='black', lw=2, marker='o', mfc='black', ms=5)
            plt.rc('text', usetex=self.usetex)
            dataaxis['label'] = ax.text(i, axmin, '\n' + self.xlabels[i],
                ha='center', va='top')
            plt.rc('text', usetex=True)
            dataaxis['min'] = ax.text(i, axmin,
                r'\hspace{4mm}' + str(xmin[i] + axmin * (xmax[i] - xmin[i])),
                ha='left', va='center')
            dataaxis['max'] = ax.text(i, axmax,
                r'\hspace{4mm}' + str(xmin[i] + axmax * (xmax[i] - xmin[i])),
                ha='left', va='center')
            dataaxes.append(dataaxis)
        plots['axes'] = dataaxes

        # Data points
        dataplots = []
        Xn = (active_X - np.tile(xmin, (xnum, 1))) / np.tile(xmax - xmin, (xnum, 1))
        ind = (m.Y == 0).flatten()
        dataplots.append(ax.plot(np.arange(0, xdim).T, Xn[ind,:].T, linestyle='-',
            color='blue', marker='o', mfc='blue', ms=2, mec='blue'))
        ind = (m.Y == 1).flatten()
        dataplots.append(ax.plot(np.arange(0, xdim).T, Xn[ind,:].T, linestyle='-',
            color='red',  marker='o', mfc='red',  ms=2, mec='red'))
        plots['data'] = dataplots

        # Latent function: TODO
        # What's a good way?

        self.fig = fig
        return plots


def getFrame(X, res=default_res, grid=True):
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

    if grid:
        xgrd = np.meshgrid(*tuple(xrng[:,i] for i in range(xdim)), indexing='ij')
        xgrd = np.hstack(tuple(xgrd[i].reshape(-1,1) for i in range(xdim)))
        return xmin, xmax, xrng, xgrd
    else:
        return xmin, xmax, xrng


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
            color=fillcolor,
            zorder=-999)

    return plots
