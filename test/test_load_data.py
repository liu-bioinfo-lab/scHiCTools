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
               max_distance=4000000, format='.hic', adjust_resolution=True,
               chromosomes='all', operations=['convolution'], kernel_shape=3,
               keep_n_strata=10, store_full_map=True)
    assert x.num_of_cells==len(files)


def test_readtxt():
    files = ['data/cell_03','data/cell_01','data/cell_02']
    x = scHiCs(files, reference_genome='mm9', resolution=100000,
               max_distance=4000000, format='shortest_score',
               adjust_resolution=False, chromosomes='except Y',
                operations=['convolution', 'random_walk', 'network_enhancing'], kernel_shape=3, keep_n_strata=10)
    
    y = scHiCs(files, reference_genome='mm9', resolution=100000,
               max_distance=4000000, format='shortest_score',
               adjust_resolution=False, chromosomes='except Y',
                operations=['convolution', 'random_walk', 'network_enhancing'], kernel_shape=3, keep_n_strata=10,
               store_full_map=True, parallelize=True)
    assert x.num_of_cells==len(files)
    assert y.num_of_cells==len(files)
    assert x.resolution == y.resolution
    assert x.chromosomes == y.chromosomes
    assert x.chromosome_lengths == y.chromosome_lengths
    assert np.array_equal(x.contacts, y.contacts)
    assert np.array_equal(x.short_range,y.short_range)
    assert np.array_equal(x.mitotic,y.mitotic)


def test_readmcool():
    files=['data/test.mcool']
    y = scHiCs(files, reference_genome='hg19', resolution=2500000,
                max_distance=4000000, format='cool', adjust_resolution=True,
                chromosomes='all', operations=['convolution'],
                kernel_shape=3, keep_n_strata=10, store_full_map=True)
    assert y.num_of_cells==len(files)

def test_select_cells():
    files = ['data/cell_03','data/cell_01','data/cell_02']
    x = scHiCs(files, reference_genome='mm9', resolution=50000,
               max_distance=4000000, format='shortest_score',
               adjust_resolution=True, chromosomes='except Y',
               operations=['convolution'], kernel_shape=3, keep_n_strata=10,
               store_full_map=False)
    x.plot_contacts()
    x.select_cells(n_contacts=[240000,float('inf')], short_range=[0,0.583], mitotic=[0,.9])
    assert (x.contacts>240000).all()
    assert (x.short_range/x.contacts<=0.583).all()
    assert (x.mitotic/x.contacts<=.9).all()

