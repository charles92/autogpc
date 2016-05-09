# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
from sklearn.cross_validation import KFold

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

        self.X, self.Y = X, Y

        # Default X, Y labels
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


    def getClass(self, y):
        """
        Get data points of a specific class.

        :param y: class label
        :returns: array of data points whose class is `y`
        """
        return self.X[self.Y[:,0] == y]


    def getDataShape(self):
        xmu = self.X.mean(axis=0).flatten().tolist()
        xsd = self.X.std(axis=0).flatten().tolist()
        xmin = self.X.min(axis=0).flatten().tolist()
        xmax = self.X.max(axis=0).flatten().tolist()
        ysd = self.Y.std()
        return {
            'x_mu':  xmu,
            'x_sd':  xsd,
            'x_min': xmin,
            'x_max': xmax,
            'y_sd':  ysd
        }


    def kFoldSplits(self, k=5):
        """
        Split dataset into training sets and validation sets for k-fold
        cross-validation. When k = 1 the dataset is not partitioned -- the
        entire set is used for both training and testing.
        Result is cached after first called, unless k changes.

        :param k: desired number of blocks after the split
        :returns: tuple(X, Y, XT, YT), where T stands for 'test' and each entry
        is a list of k numpy arrays
        """
        assert k > 0 and k <= self.getNum(), "Invalid number of folds"

        if hasattr(self, 'splits') and hasattr(self, 'nFolds') and self.nFolds == k:
            return self.splits

        else:
            if k == 1:
                ret = [self.X], [self.Y], [self.X], [self.Y]
            else:
                X, Y, XT, YT = [], [], [], []
                kf = KFold(self.getNum(), n_folds=k, shuffle=True)
                for train_ind, test_ind in kf:
                    X.append(self.X[train_ind])
                    Y.append(self.Y[train_ind])
                    XT.append(self.X[test_ind])
                    YT.append(self.Y[test_ind])
                ret = X, Y, XT, YT

            self.splits = ret
            self.nFolds = k
            return ret

