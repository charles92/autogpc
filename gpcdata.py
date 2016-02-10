# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np

class GPCData(object):
	"""
	Dataset for AutoGPC
	"""
	def __init__(X, Y, XLabel=None, YLabel=None):
		# Sanity check
		assert isinstance(X, np.ndarray), "X must be a Numpy ndarray"
		assert X.ndim == 2, "X must be a two-dimensional array"
		assert isinstance(Y, np.ndarray), "Y must be a Numpy ndarray"
		assert Y.ndim == 1, "Y must be a vector"
		assert Y.size[0] == X.size[0], "X and Y must contain the same number of entries"

		self.X = X
		self.Y = Y

		if XLabel is not None and isinstance(XLabel, list) and len(XLabel) == X.size[1]:
			self.XLabel = XLabel
		else:
			self.XLabel = ['$x_{{{0}}}$'.format(d) for d in range(1, X.size[1])]

		if YLabel is not None and isinstance(YLabel, basestring):
			self.YLabel = YLabel
		else:
			self.YLabel = '$y$'
