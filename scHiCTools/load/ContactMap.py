"""
Initializing scHiC contact map of one cell.
"""

from scipy.sparse import coo_matrix
import numpy as np
from .load_hic_file import *


class scHiC:
    def __init__(self, matrices=None, file=None, format=None, sparse=True,
                 reference_genome=None, resolution=None, **kwargs):
        """
        :param matrices:
        :param format:
        :param sparse:
        :param file:
        :param resolution:
        :param
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
            elif format in ['hic', 'HIC']:
                self.chromosomes, self.chromosomes_lengths, self.maps =\
                    ContactMap_from_HiC(file, reference_genome, resolution, sparse, **kwargs)

            elif format == 'sparse_matrix':
                # Need repair
                pass

            elif format == 'customized':
                self.chromosomes, self.chromosomes_lengths, self.maps =\
                    ContactMap_from_text(file, reference_genome, resolution, sparse, **kwargs)

            else:
                raise ValueError('Input format {0} not supported. Only "hic", "sparse_matrix" or "customized".')
        self.processed_maps = {k: None for k in self.chromosomes}

    def encode(self):
        if self.sparse:
            raise ValueError('The matrices have already been encoded!')
        for k in self.maps.keys():
            self.maps[k] = coo_matrix(self.maps[k])
        self.sparse = True

    def decode(self):
        if not self.sparse:
            raise ValueError('The matrices have already been decoded!')
        for k in self.maps.keys():
            self.maps[k] = self.maps[k].toarray()
        self.sparse = False

    def add_chromosome(self, name, mat):
        if name in self.maps:
            print('Warning! {0} would be replaced!'.format(name))
        self.maps[name] = mat

    def remove_chromosome(self, name):
        if name not in self.chromosomes:
            print('Warning! {0} not one of the existing chromosomes.'.format(name))
        else:
            del self.chromosomes_lengths[name]
            del self.maps[name]
            del self.processed_maps[name]
            self.chromosomes.remove(name)

