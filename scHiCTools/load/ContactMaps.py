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
        self.resolution = resolution
        self.num_of_cells = len(list_of_files)

        for file in list_of_files:
            print('Processing in file: {0}'.format(file))
            cell = scHiC(file=file, format=format, resolution=resolution, reference_genome=reference_genome,
                         sparse=False, **kwargs)

            if normalization is not None:
                normalization_cell(cell, normalization)

            if preprocessing is not None:
                processing(cell, preprocessing, **kwargs)

            # Get stripes
            if max_distance is None:
                self.store_full_map = True
                self.sparse = sparse
                for chromosome_name in cell.chromosomes:
                    if chromosome_name not in self.stripes:
                        self.stripes[chromosome_name] = []
                    if cell.processed_maps[chromosome_name] is None:
                        m = cell.maps[chromosome_name]
                    else:
                        m = cell.processed_maps[chromosome_name]
                    if self.sparse:
                        m = coo_matrix(m)
                    self.stripes[chromosome_name].append(m)
            else:
                self.store_full_map = False
                self.sparse = False
                num_of_stripes = max_distance // resolution
                for chromosome_name in cell.chromosomes:
                    if chromosome_name not in self.stripes:
                        self.stripes[chromosome_name] = []
                    if cell.processed_maps[chromosome_name] is None:
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
        for chromosome_name in self.stripes:
            for idx, stripes in enumerate(self.stripes[chromosome_name]):
                if self.store_full_map:
                    m = stripes
                    if self.sparse:
                        m = m.toarray()
                else:  # Reconstruct full map
                    m = np.zeros((len(stripes[0]), len(stripes[0])))
                    for i, stripe in enumerate(stripes):
                        for j in range(len(stripe)):
                            m[j, i+j] = stripe[j]
                            if i != 0:
                                m[i+j, j] = stripe[j]

                for method in methods:
                    if method == 'convolution':
                        m = convolution(m, **kwargs)
                    elif method == 'random_walk':
                        m = random_walk(m, **kwargs)
                    elif method == 'reduce_sparsity':
                        m = reduce_sparsity(m, **kwargs)
                    elif method == 'network_enhancing':
                        m = network_enhancing(m, **kwargs)
                    elif method == 'normalization':
                        m = normalization_matrix(m, method)
                    else:
                        print('Operation not in [reduce_sparsity, convolution, random_walk, network_enhancing， normalization].\
                                         Operation omitted.')

                if self.store_full_map:
                    if self.sparse:
                        new_stripes = coo_matrix(m)
                    else:
                        new_stripes = m
                else:
                    new_stripes = [np.diag(m[i:, :len(m)-i]) for i in range(len(stripes))]
                self.stripes[chromosome_name][idx] = new_stripes

    def learn_embedding(self, method, dim=2, aggregation='median', max_distance=None, return_distance=False, **kwargs):
        """
        Learn a low-dimensional embedding for cells.
        :param method: (str) 'inner_product', 'HiCRep' or 'Selfish'
        :param aggregation: (str) 'mean' or 'median'
        :param dim: (int) dimension of the embedding
        :param max_distance: (int or None) max_distance: only consider contacts within this genomic distance, default: None.
        If None, it will use the 'max_distance' in previous loading data process, thus if you
        set 'max_distance=None' in data loading, you must specify a max distance for this step
        :param return_distance: (bool) if True, return (embeddings, distance_matrix); if False, only return embeddings

        Some additional argument for Selfish:
          :param n_windows: number of Selfish windows
          :param sigma: sigma in the Gaussian-like kernel

        :return: embeddings: numpy.array (shape: num_of_cells * dimension),
        distance_matrix: numpy.array (shape: num_of_cells * num_of_cells)
        """
        distance_matrices = []
        num_stripes = max_distance // self.resolution if max_distance is not None else -1
        for chromosome_name in self.chromosomes:
            print('Calculating pairwise distances for {0}'.format(chromosome_name))
            if self.store_full_map:
                if max_distance is None:
                    raise ValueError('You must specify max genomic distance to use!')
                full_maps = self.stripes[chromosome_name]
                stripes = []
                for fmap in full_maps:
                    if self.sparse:
                        fmap = fmap.toarray()
                    new_stripes = [np.diag(fmap[i:, :len(fmap) - i]) for i in range(len(num_stripes))]
                    stripes.append(new_stripes)
            else:
                stripes = self.stripes[chromosome_name]
                if max_distance is None or num_stripes >= len(stripes[0]):
                    pass
                else:
                    stripes = [new_stripes[:num_stripes] for new_stripes in stripes]
            distance_matrix = pairwise_distances(stripes, method)
            distance_matrices.append(distance_matrix)
        distance_matrices = np.array(distance_matrices)

        if aggregation == 'mean':
            final_distance = np.mean(distance_matrices, axis=0)
        elif aggregation == 'median':
            final_distance = np.median(distance_matrices, axis=0)
        else:
            raise ValueError('Aggregation method {0} not supported. Only "mean" or "median".'.format(aggregation))

        embeddings = MDS(final_distance, dim)

        if return_distance:
            return embeddings, final_distance
        else:
            return embeddings


