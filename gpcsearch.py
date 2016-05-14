# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import Queue as Q
import flexible_function as ff
from gpckernel import GPCKernel
from gpcdata import GPCData


class GPCSearch(object):
    """
    AutoGPC kernel search
    """
    def __init__(self, data=None, base_kernels='SE', max_depth=3, beam_width=1):
        assert isinstance(data, GPCData), "data must be of type GPCData"
        self.data = data
        # TODO: sanity check for base_kernels
        self.baseKernels = base_kernels
        # TODO: sanity check for max_depth
        self.maxDepth = max_depth
        # TODO: sanity check for beam_width
        self.beamWidth = beam_width

    def search(self):
        # Start from NoneKernel
        depth = 0
        kernels = [GPCKernel(ff.NoneKernel(), self.data, depth=depth)]
        best = [kernels[0]]

        print "\n=====\nSearch begins:"
        while depth < self.maxDepth:
            newkernels = []
            print "\n=====\nExpanding:"
            for k in kernels:
                print k
                # Enforce new dimensions to be added each time
                expanded = [x for x in k.expand() if len(x.getActiveDims()) > depth]
                newkernels.extend(expanded)
            if depth == 0:
                kernels1d = newkernels

            kernels = newkernels
            for k in kernels:
                k.train()
            print "\n=====\nFully expanded kernel set at depth {0}:".format(depth+1)
            print '\n'.join(str(k) for k in kernels)

            kernels = sorted(kernels, key=lambda x: x.error())
            if len(kernels) == 0 or kernels[0].error() >= best[-1].error():
                break

            best.append(kernels[0])
            if len(kernels) > self.beamWidth:
                kernels = kernels[:self.beamWidth]
            depth = depth + 1

        # In case an additive component is 1-D, and performs better than the
        # initial 1-D kernel (this is possible due to random initialisation,
        # etc.), we replace that 1-D kernel with the one found in additive
        # components.
        summands = best[-1].toSummands()
        summands1d = filter(lambda k: len(k.getActiveDims()) == 1, summands)
        for k in summands1d:
            for i in range(len(kernels1d)):
                if k.equals(kernels1d[i]) and k.betterThan(kernels1d[i]):
                    kernels1d[i] = k
                    break
        best1d = bestKernels1D(kernels1d)

        # In case the best search result is just a 1-D kernel, and it performs
        # worse than the kernel in `best1d`, we replace the one in `best` and
        # `summands` with this better kernel
        if len(best[-1].getActiveDims()) == 1:
            for k in best1d:
                if k.equals(best[-1]) and k.betterThan(best[-1]):
                    best[-1] = k
                    summands[0] = k
                    break

        print "\n=====\nSearch completed. Best kernels at each depth:"
        for k in best:
            print k

        print "\n=====\nSummands:"
        for k in summands:
            print k

        print "\n=====\nBest 1-D kernels:"
        for k in best1d:
            print k

        return best, best1d


    def baseline(self):
        """
        Train the baseline model (constant kernel):
        `flexible_function.ConstKernel` in GPSS
        `GPy.kern.Bias` in GPy

        :returns: trained `GPCKernel` object which uses a constant kernel
        """
        k = GPCKernel(ff.ConstKernel(), self.data)
        k.train()
        return k


##############################################
#                                            #
#             Helper Functions               #
#                                            #
##############################################

def bestKernels1D(kernels):
    """
    Select the best 1-D kernels in each dimension according to cross-validated
    training error rate.

    :param kernels: list of 1-D kernels of type `GPCKernel`
    :returns: list of best 1-D kernels, ranked by cross-validated training error
    """
    if len(kernels) == 0: return []
    assert all([len(k.getActiveDims()) == 1 for k in kernels]), 'All kernels must be one-dimensional.'
    data = kernels[0].data
    ndim = data.getDim()

    # Initialise
    best1d = [GPCKernel(ff.NoneKernel(), data) for i in range(ndim)]

    # Run through all candidates
    for k in kernels:
        dim = k.getActiveDims()[0]
        if k.error() < best1d[dim].error():
            best1d[dim] = k

    # Rank by cross-validated training error
    best1d.sort(key=lambda k: k.error())

    return best1d
