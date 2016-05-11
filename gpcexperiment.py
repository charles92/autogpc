# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pods
from gpcdata import GPCData
from gpcreport import GPCReport
from gpcsearch import GPCSearch
import time

# Simple timer from
# http://stackoverflow.com/questions/5478351/python-time-measure-function
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f seconds.' % (f.func_name, time2-time1)
        return ret
    return wrap


class GPCExperiment(object):
    """
    Experiments for AutoGPC
    """
    def __init__(self):
        pass

    @timing
    def pima(self, dims=[1,5,6,7], depth=4, width=2):
        data = pods.datasets.pima()
        X, Y = data['X'][:,dims], data['Y']
        XLabel = data['XLabel']
        d = GPCData(X, Y, XLabel=XLabel, YLabel=['not diabetic', 'diabetic'])
        print "\n\nPima Indian Diabetes dataset"
        print "=====\nData size: D = %d, N = %d." % (d.getDim(), d.getNum())
        search = GPCSearch(data=d, max_depth=depth, beam_width=width)
        best, best1d = search.search()
        report = GPCReport(name='Pima', history=best, best1d=best1d)
        report.export()
        print ""


    @timing
    def wisconsin(self, dims=[0,1,4,5,7], depth=4, width=2):
        data = pods.datasets.breastoriginal()
        X, Y = data['X'][:,dims], data['Y']
        XLabel = data['XLabel']
        d = GPCData(X, Y, XLabel=XLabel, YLabel=['no breast cancer', 'with breast cancer'])
        print "\n\nWisconsin Breast Cancer dataset"
        print "=====\nData size: D = %d, N = %d." % (d.getDim(), d.getNum())
        search = GPCSearch(data=d, max_depth=depth, beam_width=width)
        best, best1d = search.search()
        report = GPCReport(name='Wisconsin', history=best, best1d=best1d)
        report.export()


    @timing
    def bupa(self, dims=range(5), depth=4, width=2):
        data = pods.datasets.bupa()
        X, Y = data['X'][:,dims], data['Y']
        XLabel = data['XLabel']
        d = GPCData(X, Y, XLabel=XLabel, YLabel=['$\\leq 5$ drink units', '$> 5$ drink units'])
        print "\n\nBUPA Liver Disorders dataset"
        print "=====\nData size: D = %d, N = %d." % (d.getDim(), d.getNum())
        search = GPCSearch(data=d, max_depth=depth, beam_width=width)
        best, best1d = search.search()
        report = GPCReport(name='BUPA', history=best, best1d=best1d)
        report.export()


    @timing
    def cleveland(self, dims=range(13), depth=5, width=2):
        data = pods.datasets.cleveland()
        X, Y = data['X'][:,dims], data['Y']
        XLabel = data['XLabel']
        d = GPCData(X, Y, XLabel=XLabel, YLabel=['no heart disease', 'with heart disease'])
        print "\n\nCleveland Heart Disease dataset"
        print "=====\nData size: D = %d, N = %d." % (d.getDim(), d.getNum())
        search = GPCSearch(data=d, max_depth=4, beam_width=2)
        best, best1d = search.search()
        report = GPCReport(name='Cleveland', history=best, best1d=best1d)
        report.export()


##############################################
#                                            #
#             Helper Functions               #
#                                            #
##############################################


