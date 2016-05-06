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

            kernels = newkernels
            for k in kernels:
                k.train()
            print "\n=====\nFully expanded kernel set at depth {0}:".format(depth+1)
            print '\n'.join(k.__repr__() for k in kernels)

            kernels = sorted(kernels, key=lambda x: x.getCvError())
            if len(kernels) == 0 or kernels[0].getCvError() >= best[-1].getCvError():
                break

            best.append(kernels[0])
            if len(kernels) > self.beamWidth:
                kernels = kernels[:self.beamWidth]
            depth = depth + 1

        print "\n=====\nSearch completed. Best kernels at each depth:"
        for k in best:
            print k
        print "\n=====\nBack-tracking:"
        k = best[-1]
        while k.depth > 0:
            print k
            k = k.parent
        return best
