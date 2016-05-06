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
            print '\n\ndepth={0}'.format(depth)
            newkernels = []
            for k in kernels:
                print "\n=====\nExpanding:"
                print k
                expanded = k.expand()
                # print "  Expanded kernels:"
                # print '\n'.join(k.__repr__() for k in expanded)
                newkernels.extend(expanded)

            print "\n=====\nFully expanded kernel set:"
            kernels = newkernels
            for k in kernels:
                k.train()
            print '\n'.join(k.__repr__() for k in kernels)

            kernels = sorted(kernels, key=lambda x: x.getNLML())
            if len(kernels) == 0 or kernels[0].getNLML() > best[-1].getNLML():
                break

            best.append(kernels[0])
            if len(kernels) > self.beamWidth:
                kernels = kernels[:self.beamWidth]
            depth = depth + 1

        return best
