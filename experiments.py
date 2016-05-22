# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

from gpcexperiment import GPCExperiment

ex = GPCExperiment()

for i in range(3):
    ex.bupa()
    ex.pima()
    ex.wisconsin()
    ex.cleveland()
