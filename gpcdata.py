# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np

class GPCData(object):
	"""
	Dataset for AutoGPC
	"""
	def __init__(self, X, Y, XLabel=None, YLabel=None):
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
			self.XLabel = XLabel
		else:
			self.XLabel = ['$x_{{{0}}}$'.format(d + 1) for d in range(X.shape[1])]

		if YLabel is not None and isinstance(YLabel, basestring):
			self.YLabel = YLabel
		else:
			self.YLabel = '$y$'

	def __repr__(self):
		return 'GPCData: %d dimensions, %d data points.\n' % \
		       (self.X.shape[1], self.X.shape[0]) + \
		       'XLabel:\n' + \
		       ', '.join(self.XLabel) + '\n' + \
		       'YLabel:\n' + \
		       self.YLabel
