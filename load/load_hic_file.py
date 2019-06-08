"""
This file is for loading a single HiC contact map from different type of file.
"""

from scipy.sparse import coo_matrix
import numpy as np
import gzip
import os.path
from .read_hic_format import straw

my_path = os.path.abspath(os.path.dirname(__file__))


def get_chromosome_lengths(ref, res=1):
    """
    Get lengths for all chromosomes in a given resolution according to the reference genome.
    :param ref: (str) name of reference genome, eg.'mm10', 'hg38'
    :param res: (int) resolution
    :return: dict (str:int), eg. {'chr1': 395, 'chr2': 390, ...}
    """
    try:
        path = os.path.join(my_path, 'reference_genome/' + ref)
        # print(path)
        rg = open(path)
    except FileNotFoundError:
        try:
            path = os.path.join(my_path, 'reference_genome\\' + ref)
            rg = open(path)
        except FileNotFoundError:
            raise FileNotFoundError('Reference genome {0} not supported!'.format(ref))

    def _res_change(length, res):
        if length % res == 0:
            return length // res
        else:
            return length // res + 1

    rg = [line.strip().split() for line in rg]
    lengths = {lst[0]: _res_change(int(lst[1]), res) for lst in rg}
    chromosomes = set(lengths.keys())
    return chromosomes, lengths


def get_chromosomes(all_chromosomes, **kwargs):
    """
    According to the **kwargs given, decide which chromosomes to use.
    :param all_chromosomes: set of all chromosomes
    :param chromosomes: 'All', 'no X', 'no Y' or a list of chromosomes like ['chr1', 'chr2']
    :return: set of chromosomes to use
    """
    if 'chromosomes' not in kwargs:
        print('Did not specify which chromosomes to use. Use default: chr1-19 + chrX + chrY')
    else:
        if kwargs['chromosomes'] in ['all', 'All']:
            pass
        elif kwargs['chromosomes'] == 'no X':
            all_chromosomes.remove('chrX')
        elif kwargs['chromosomes'] == 'no Y':
            all_chromosomes.remove('chrY')
        elif kwargs['chromosomes'] == 'no XY':
            all_chromosomes.remove('chrX')
            all_chromosomes.remove('chrY')
        elif type(kwargs['chromosomes']) == list:
            all_chromosomes = set(kwargs['chromosomes'])
    return all_chromosomes


def re_order_line(line, order):
    if type(order) == int:
        order = [int(elm) for elm in str(order)]

    lst = line.strip().split()
    chr1, pos1, chr2, pos2 = [lst[i - 1] for i in order[:4]]
    if len(order) == 5:
        v = lst[order[4] - 1]
    else:
        v = 1
    # Optimize this part!
    if 'chr' not in chr1:
        chr1 = 'chr' + chr1
    if 'chr' not in chr2:
        chr2 = 'chr' + chr2
    return chr1, pos1, chr2, pos2, v


def ContactMap_from_text(file, reference_genome, resolution, sparse, **kwargs):
    chromosomes, lengths = get_chromosome_lengths(reference_genome, resolution)
    chromosomes = get_chromosomes(**kwargs)
    maps = {k: np.zeros((lengths[k], lengths[k])) for k in chromosomes}

    if 'line_format' not in kwargs:
        raise ValueError('Format in each line must be provided while using customized format.')
    order = kwargs['line_format']

    if 'header' not in kwargs:
        print(' Did not specify whether to skip header. Use default: False.')
        header = False
    else:
        header = kwargs['header']

    if 'gzip' not in kwargs:
        gz = False
    else:
        gz = kwargs['gzip']
    open_ = gzip.open if gz else open

    if 'adjust_resolution' not in kwargs:
        print(' Did not specify whether to change resolution. Use default: True.')
        adjust_resolution = True
    else:
        adjust_resolution = kwargs['adjust_resolution']

    for line in open_(file):
        chr1, pos1, chr2, pos2, v = re_order_line(line, order)
        if chr1 != chr2:
            continue
        if adjust_resolution:
            pos1, pos2 = int(pos1) // resolution, int(pos2) // resolution
        maps[chr1][pos1, pos2] += v
        if pos1 != pos2:
            maps[chr1][pos2, pos1] += v
    if sparse:
        for k in maps.keys():
            maps[k] = coo_matrix(maps[k])
    return chromosomes, lengths, maps


def ContactMap_from_HiC(file, reference_genome, resolution, sparse, **kwargs):
    chromosomes, lengths = get_chromosome_lengths(reference_genome, resolution)
    chromosomes = get_chromosomes(**kwargs)
    maps = {k: np.zeros((lengths[k], lengths[k])) for k in chromosomes}
    for chromosome in chromosomes:
        xs, ys, counts = straw('NONE', file, chromosome, chromosome, 'BP', resolution)
        for x, y, count in zip(xs, ys, counts):
            x, y = x // resolution, y // resolution
            maps[chromosome][x, y] += count
            if x != y:
                maps[chromosome][y, x] += count
    if sparse:
        for k in maps.keys():
            maps[k] = coo_matrix(maps[k])
    return chromosomes, lengths, maps
