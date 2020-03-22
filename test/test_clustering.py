# -*- coding: utf-8 -*-

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..')))


from scHiCTools import kmeans, spectral_clustering, HAC


center=np.array([[0,0],[20,20],[-20,20]])
random_generate_data = np.random.normal(size=(60,2)) + np.repeat(center, 20, axis=0)

def test_kmeans():
    label=kmeans(random_generate_data, k=3)
    l1 = np.equal(label[:20], label[0]).all()
    l2 = np.equal(label[20:-20], label[20]).all()
    l3 = np.equal(label[-20:], label[-20]).all()
    assert l1
    assert l2
    assert l3


def test_SC():
    label=spectral_clustering(random_generate_data, n_clusters=3)
    l1 = np.equal(label[:20], label[0]).all()
    l2 = np.equal(label[20:-20], label[20]).all()
    l3 = np.equal(label[-20:], label[-20]).all()
    assert l1
    assert l2 
    assert l3
    
def test_HAC():
    label=HAC(random_generate_data, n_clusters=3)
    l1 = np.equal(label[:20], label[0]).all()
    l2 = np.equal(label[20:-20], label[20]).all()
    l3 = np.equal(label[-20:], label[-20]).all()
    assert l1
    assert l2
    assert l3
    



