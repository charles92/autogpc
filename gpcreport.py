# Copyright (c) 2015, Qiurui He
# Department of Engineering, University of Cambridge

import numpy as np
import pylatex as pl
import pylatex.utils as ut

class GPCReport(object):
    """
    AutoGPC data analysis report.
    """

    def __init__(self, paper='a4paper'):
        self.doc = pl.Document()
        self.makePreamble(paper=paper)

    def makePreamble(self, paper='a4paper'):
        doc = self.doc
        doc.packages.append(pl.Package('geometry', options=['a4paper', 'margin=1.5in']))
        doc.preamble.append(pl.Command('title', 'AutoGPC Data Analysis Report'))
        doc.preamble.append(pl.Command('author', 'Automatic Statistician'))
        doc.preamble.append(pl.Command('date', ut.NoEscape(r'\today')))
        doc.append(ut.NoEscape(r'\maketitle'))

    def export(self, filename=None):
        if filename is None:
            filename = r'./latex/report'
        self.doc.generate_pdf(filename)
