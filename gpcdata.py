# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np

class GPCData(object):
	"""
	Dataset for AutoGPC
	"""
	def __init__(self, X, Y, XLabel=None, YLabel=None):
		"""
		Instantiate a dataset for AutoGPC with N data points and D input
		dimensions.

		:param X: NxD matrix of real-valued inputs
		:param Y: Nx1 matrix of {0,1}-valued class labels
		:param XLabel: D-tuple or D-list of axis labels
		:param YLable: y axis label string
		"""
		# Sanity check
		assert isinstance(X, np.ndarray), "X must be a Numpy ndarray"
		assert X.ndim == 2, "X must be a two-dimensional array"
		assert isinstance(Y, np.ndarray), "Y must be a Numpy ndarray"
		assert Y.ndim == 2 and Y.shape[1] == 1, "Y must be a vector"
		assert Y.shape[0] == X.shape[0], "X and Y must contain the same number of entries"

		# Populate instance fields
		self.X = X
		self.Y = Y

		if XLabel is not None and isinstance(XLabel, (list,tuple)) and len(XLabel) == X.shape[1]:
			self.XLabel = tuple(XLabel)
		else:
			self.XLabel = tuple('$x_{{{0}}}$'.format(d + 1) for d in range(X.shape[1]))

		if YLabel is not None and isinstance(YLabel, basestring):
			self.YLabel = YLabel
		else:
			self.YLabel = '$y$'

	def __repr__(self):
		return 'GPCData: %d dimensions, %d data points.\n' % \
		       (self.getDim(), self.getNum()) + \
		       'XLabel:\n' + \
		       ', '.join(self.XLabel) + '\n' + \
		       'YLabel:\n' + \
		       self.YLabel

	def getNum(self):
		return self.X.shape[0]

	def getDim(self):
		return self.X.shape[1]

	def getDataShape(self):
		xsd = np.std(self.X, axis=0).flatten().tolist()
		xmin = np.amin(self.X, axis=0).flatten().tolist()
		xmax = np.amax(self.X, axis=0).flatten().tolist()
		ysd = np.std(self.Y)
		return {
			'x_sd':  xsd,
			'x_min': xmin,
			'x_max': xmax,
			'y_sd':  ysd
		}

