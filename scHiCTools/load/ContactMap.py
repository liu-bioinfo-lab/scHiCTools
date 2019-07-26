"""
Initializing scHiC contact map of one cell.
"""

from scipy.sparse import coo_matrix
import numpy as np
from .load_hic_file import *


class scHiC:
    def __init__(self, sparse=True, matrices=None, file=None, format=None,
                 reference_genome=None, resolution=None, **kwargs):
        """
        :param sparse: (bool) whether to use sparse matrix to store, default: True
        :param matrices: (dict) contact maps can be directly given in format dict: {chromosome name (str): contact map (numpy.array)}
        If matrices given, all following arguments can be ignored.
        :param format: (str) '.hic' or 'customized'
        :param file: (str) scHiC file path
        :param resolution: (int) resolution
        Other arguments:
        :param header: (bool) whether the files have a header line
        :param gzip: (bool) whether the file is in gzip format
        :param line_format: (str or list) the column indices in the order of chromosome 1 - position 1 - chromosome 2 - position 2 - read count.
        e.g. if one line is "chr1 5000000 chr2 3500000 2", the format is '12345' or [1, 2, 3, 4, 5]; if there is no
        number of reads in the line, you can just provide '1234' and read will be set to default value 1. Default: '12345'
        :param chromosomes: chromosomes to use, eg. ['chr1', 'chr2'], or just 'except Y', 'except XY', 'all', default: 'all', which means
        chr 1-19 + XY for mouse and chr 1-22 + XY for human
        :param adjust_resolution: whether to adjust resolution for input file. Sometimes the input file is already
        in the proper resolution (e.g. position 3000000 has already been changed to 6 in 500kb resolution), then
        you can set adjust_resolution=False. Default: True
        """
        self.sparse = sparse
        if matrices is not None:
            if type(matrices) != dict:
                raise TypeError('Contact map must be given as a dictionary {chromosome (str): contact map (numpy.array)}')
            self.chromosomes = set(matrices.keys())
            self.chromosomes_lengths = {k: len(v) for k, v in matrices.items()}
            if self.sparse:
                self.maps = {k: coo_matrix(v) for k, v in matrices.items()}
            else:
                self.maps = {k: np.array(v) for k, v in matrices.items()}
        else:
            if file is None:
                raise ValueError('Input file must be provided!')
            if format is None:
                raise ValueError('Input format must be provided!')
            elif format in ['.hic', '.HIC']:
                self.chromosomes, self.chromosomes_lengths, self.maps =\
                    ContactMap_from_HiC(file, reference_genome, resolution, sparse, **kwargs)

            elif format == 'customized':
                self.chromosomes, self.chromosomes_lengths, self.maps =\
                    ContactMap_from_text(file, reference_genome, resolution, sparse, **kwargs)

            else:
                raise ValueError('Input format {0} not supported. Only ".hic" or "customized".')
        self.processed_maps = {k: None for k in self.chromosomes}

    def encode(self):
        # Encode from full matrix to sparse matrix
        if self.sparse:
            raise ValueError('The matrices have already been encoded!')
        for k in self.maps.keys():
            self.maps[k] = coo_matrix(self.maps[k])
        self.sparse = True

    def decode(self):
        # Decode from sparse matrix to full matrix
        if not self.sparse:
            raise ValueError('The matrices have already been decoded!')
        for k in self.maps.keys():
            self.maps[k] = self.maps[k].toarray()
        self.sparse = False

    def add_chromosome(self, name, mat):
        # Currently not used
        if name in self.maps:
            print('Warning! {0} would be replaced!'.format(name))
        self.maps[name] = mat

    def remove_chromosome(self, name):
        # Currently not used
        if name not in self.chromosomes:
            print('Warning! {0} not one of the existing chromosomes.'.format(name))
        else:
            del self.chromosomes_lengths[name]
            del self.maps[name]
            del self.processed_maps[name]
            self.chromosomes.remove(name)

