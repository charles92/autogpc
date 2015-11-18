# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

default_dpi = 1200

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

    def show(self):
        print 'DEBUG: GPCPlot show():'
        print self.model
        plt.close('all')
        self.draw()
        plt.show()

    def save(self, fname):
        print 'DEBUG: GPCPlot save()'
        print self.model
        plt.close('all')
        self.draw()
        plt.savefig(fname=fname, dpi=self.dpi, format='eps')


class GPCPlot1D(GPCPlot):
    """
    Gaussian process classification plot: 1-dimensional input

    """

    def __init__(self, model, dpi=default_dpi):
        GPCPlot.__init__(self, model, dpi)

    def draw(self):
        print 'TODO: 1D'


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
