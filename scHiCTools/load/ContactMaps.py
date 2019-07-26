import numpy as np
from scipy.sparse import coo_matrix
from ..functions import *
from .ContactMap import scHiC
from ..embedding import *


class scHiCs:
    def __init__(self, list_of_files, reference_genome, resolution, sparse=True,
                 format='customized', max_distance=None, normalization=None,
                 preprocessing=None, **kwargs):
        """
        :param list_of_files (list): list of HiC file paths
        :param reference genome: now supporting 'mm9', 'mm10', 'hg19', 'hg38',
        if using other references, you can simply provide the chromosome name and
        corresponding size (bp) with a dictionary in Python.
        e.g. {'chr1': 150000000, 'chr2': 130000000, 'chr3': 200000000}
        :param resolution: (int) resolution: the resolution to separate genome into bins.
        If using .hic file format, the given resolution must match with the resolutions in .hic file.
        :param sparse: (bool) whether to use sparse matrix to store (only effective when max_distance=None),
        default: True
        :param format: (str) '.hic' or 'customized', default: 'customized'
        :param max_distance: (None or int) only consider contacts within this genomic distance, default: None.
        If 'None', it will store full matrices in scipy sparse matrix format, which will use too much memory sometimes
        :param normalization: (None or str) 'OE' (observed / expected), 'VC', 'VC_SQRT', 'KR'
        :param preprocessing: (list) the methods use for pre-processing or smoothing the maps given in a list.
        The operations will happen in the given order. Operations: 'reduce_sparsity', 'convolution', 'random_walk', 'network_enhancing'.
        Default: None. For preprocessing and smoothing operations, sometimes you need additional arguments.
        You can check docstrings for preprocessing and smoothing for more information.

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

        # Only keep stripes data if max_distance is not None
        self.stripes = dict()
        self.num_of_cells = len(list_of_files)

        for file in list_of_files:
            print('Processing in file: {0}'.format(file))
            cell = scHiC(file=file, format=format, resolution=resolution, reference_genome=reference_genome,
                         sparse=False, **kwargs)
            if normalization is not None:
                if normalization not in ['OE', 'KR', 'VC', 'VC_SQRT']:
                    print("Normalization operation not in ['OE', 'KR', 'VC', 'VC_SQRT']. Normalization omitted.")
                normalization_matrix(cell, normalization)
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
                    print('Operation not in [reduce_sparsity, smooth, random_walk, network_enhancing].\
                     Operation omitted.')
            # Get stripes
            if max_distance is None:
                for chromosome_name in cell.chromosomes:
                    if chromosome_name not in self.stripes:
                        self.stripes[chromosome_name] = []
                    if cell.processed_maps is None:
                        m = cell.maps[chromosome_name]
                    else:
                        m = cell.processed_maps[chromosome_name]
                    if sparse:
                        m = coo_matrix(m)
                    self.stripes[chromosome_name].append(m)
            else:
                num_of_stripes = max_distance // resolution
                for chromosome_name in cell.chromosomes:
                    if chromosome_name not in self.stripes:
                        self.stripes[chromosome_name] = []
                    if cell.processed_maps is None:
                        m = cell.maps[chromosome_name]
                    else:
                        m = cell.processed_maps[chromosome_name]
                    stripe_of_cell = []
                    for i in range(num_of_stripes):
                        diag = np.diag(m[:len(m) - i, i:])
                        stripe_of_cell.append(diag)
                    self.stripes[chromosome_name].append(stripe_of_cell)
        self.chromosomes = set(self.stripes.keys())

    def preprocessing(self, methods, **kwargs):
        pass

    def learn_embedding(self, method, dim=2, aggregation='median', return_distance=False, **kwargs):
        """
        Learn a low-dimensional embedding for cells.
        :param method: (str) 'inner_product', 'HiCRep' or 'Selfish'
        :param aggregation: (str) 'mean' or 'median'
        :param dim: (int) dimension of the embedding
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

        embedding = MDS(final_distance, dim)

        if return_distance:
            return embedding, final_distance
        else:
            return embedding


