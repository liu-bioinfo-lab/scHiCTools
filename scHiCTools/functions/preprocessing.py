import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix


def smooth(scHiC, **kwargs):
    """
    Smoothing with a convolution kernel for all chromosomes' contact map for a cell.
    :param smoothing_parameter: kernel size = 2 * parameter + 1
    Update in scHiC.processed_maps
    """
    if 'smoothing_parameter' not in kwargs:
        h = 1
        print('While smoothing: parameter: smoothing_parameter missing. Use default value: 1')
    else:
        h = kwargs['smoothing_parameter']
    for chromosome_name in scHiC.maps.keys():
        if scHiC.processed_maps[chromosome_name] is None:
            m = scHiC.maps[chromosome_name].copy()
        else:
            m = scHiC.processed_maps[chromosome_name].copy()
        if scHiC.sparse:
            m = m.toarray()
        conv = np.ones((2 * h + 1, 2 * h + 1)) / ((2 * h + 1) ** 2)
        m = convolve2d(m, conv, 'same')
        if scHiC.sparse:
            m = coo_matrix(m)
        scHiC.processed_maps[chromosome_name] = m


def random_walk(scHiC, **kwargs):
    """
    Random walk for all chromosomes' contact map for a cell.
    Update in scHiC.processed_maps
    """
    if 'random_walk_ratio' not in kwargs:
        p = 0.9
        print('While doing random walk: parameter: random_walk_ratio missing. Use default value: 0.9')
    else:
        p = kwargs['random_walk_ratio']
    for chromosome_name in scHiC.maps.keys():
        if scHiC.processed_maps[chromosome_name] is None:
            m = scHiC.maps[chromosome_name].copy()
        else:
            m = scHiC.processed_maps[chromosome_name].copy()
        if scHiC.sparse:
            m = m.toarray()
        sm = np.sum(m, axis=1)
        sm = np.where(sm == 0, 1, sm)
        walk = np.zeros(m.shape)
        for i in range(len(m)):
            for j in range(len(m)):
                walk[i][j] = m[i][j] / sm[i]
        m = p * walk.T.dot(m).dot(walk) + (1 - p) * m
        if scHiC.sparse:
            m = coo_matrix(m)
        scHiC.processed_maps[chromosome_name] = m


def reduce_sparsity(scHiC, method='log', **kwargs):
    """
    Reducing the sparsity of original HiC contact map.
    :param method: (str) support 'log' or 'power'
    :param base: (int or float) base of logarithm (if method == 'log')
    :param power: (int or float) value of power, default 1/2, recommended [1/4, 1/2] (if method == 'power')
    Update in scHiC.processed_maps
    """
    if method == 'log':
        if 'base' in kwargs:
            base = kwargs['base']
        else:
            base = np.e
            print('While reducing sparsity with logarithm: Base of logarithm missing. Use default value: e')
        for chromosome_name in scHiC.maps.keys():
            if scHiC.processed_maps[chromosome_name] is None:
                m = scHiC.maps[chromosome_name].copy()
            else:
                m = scHiC.processed_maps[chromosome_name].copy()
            if scHiC.sparse:
                m = m.toarray()
            m = np.log(m + 1) / np.log(base)
            if scHiC.sparse:
                m = coo_matrix(m)
            scHiC.processed_maps[chromosome_name] = m
    elif method == 'power':
        if 'power' in kwargs:
            power = kwargs['power']
        else:
            power = 0.5
            print('While reducing sparsity with power function: Power number missing. Use default value: 1/2')
        for chromosome_name in scHiC.maps.keys():
            if scHiC.processed_maps[chromosome_name] is None:
                m = scHiC.maps[chromosome_name].copy()
            else:
                m = scHiC.processed_maps[chromosome_name].copy()
            if scHiC.sparse:
                m = m.toarray()
            m = np.power(m, power)
            if scHiC.sparse:
                m = coo_matrix(m)
            scHiC.processed_maps[chromosome_name] = m
    else:
        raise ValueError('Method {0} not supported while reducing sparsity.'.format(method))


def network_enhancing(scHiC, num_of_kNN=20, iteration=1, alpha=0.8):
    pass

