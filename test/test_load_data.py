# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..')))

from scHiCTools import scHiCs

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '.')))

print(sys.path[:2])


def test_readhic():
    files = ['data/test.hic']
    x = scHiCs(files, reference_genome='hg19', resolution=2500000,
               max_distance=4000000, format='.hic', resolution_adjust=True,
               chromosomes='all', operations=['convolution'], kernel_shape=3,
               keep_n_strata=10, store_full_map=False)
    assert x.num_of_cells==len(files)


def test_readtxt():
    files = ['data/cell_03','data/cell_01','data/cell_02']
    x = scHiCs(files, reference_genome='mm9', resolution=500000,
               max_distance=4000000, format='shortest_score',
               resolution_adjust=True, chromosomes='except Y',
               operations=['convolution'], kernel_shape=3, keep_n_strata=10,
               store_full_map=False)
    assert x.num_of_cells==len(files)



