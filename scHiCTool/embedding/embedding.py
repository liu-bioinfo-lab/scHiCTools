import numpy as np
from .functions import *
from ..load import scHiC
from ..processing import *


class scHiCs:
    def __init__(self, list_of_files, format, reference_genome, resolution, max_distance, preprocessing=[], **kwargs):
        # Only keep stripes data
        self.stripes = dict()
        self.num_of_cells = len(list_of_files)

        for file in list_of_files:
            print('Processing in file: {0}'.format(file))
            cell = scHiC(file=file, format=format, resolution=resolution, reference_genome=reference_genome, sparse=False, **kwargs)
            # pre_processing_options = ['reduce_sparsity', 'smooth', 'random_walk', 'network_enhancing']
            for operation in preprocessing:
                if operation == 'reduce_sparsity':
                    reduce_sparsity(cell, **kwargs)
                elif operation == 'smooth':
                    smooth(cell, **kwargs)
                elif operation == 'random_walk':
                    random_walk(cell, **kwargs)
                elif operation == 'network_enhancing':
                    network_enhancing(cell, **kwargs)
                else:
                    print('Operation not in [reduce_sparsity, smooth, random_walk, network_enhancing]. Operation omitted.')
            # Get stripes
            num_of_stripes = max_distance // resolution
            for chromosome_name in cell.chromosomes:
                if chromosome_name not in self.stripes:
                    self.stripes[chromosome_name] = []
                m = cell.maps[chromosome_name]
                stripe_of_cell = []
                for i in range(num_of_stripes):
                    diag = np.diag(m[:len(m) - i, i:])
                    stripe_of_cell.append(diag)
                self.stripes[chromosome_name].append(stripe_of_cell)
        self.chromosomes = set(self.stripes.keys())

    def learn_embedding(self, method, aggregation='median', dimension=2, return_distance=False, **kwargs):
        """
        Learn a low-dimensional embedding for cells.
        :param method: (str) 'inner_product', 'HiCRep' or 'Selfish'
        :param aggregation: (str) 'mean' or 'median'
        :param dimension: (int) dimension of the embedding
        :return: numpy.array (shape: num_of_cells * dimension)
        """
        distance_matrices = []
        for chromosome_name in self.chromosomes:
            print('Calculating pairwise distances for {0}'.format(chromosome_name))
            distance_matrix = pairwise_distances(self.stripes[chromosome_name], method, **kwargs)
            distance_matrices.append(distance_matrix)
        distance_matrices = np.array(distance_matrices)
        if aggregation == 'mean':
            final_distance = np.mean(distance_matrices, axis=0)
        elif aggregation == 'median':
            final_distance = np.median(distance_matrices, axis=0)
        else:
            raise ValueError('Aggregation method {0} not supported. Only "mean" or "median".'.format(aggregation))

        embedding = MDS(final_distance, dimension)

        if return_distance:
            return embedding, final_distance
        else:
            return embedding


