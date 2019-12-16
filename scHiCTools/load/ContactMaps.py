import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix
from .load_hic_file import get_chromosome_lengths, load_HiC
from ..embedding import pairwise_distances, MDS, tSNE, UMAP
from .processing_utils import matrix_operation


class scHiCs:
    def __init__(self, list_of_files, reference_genome, resolution, sparse=False,
                 format='customized', keep_n_strata=0, store_full_map=False,
                 operations=None, **kwargs):
        """
        Args:
            list_of_files (list): list of HiC file paths
            reference genome (str or dict): now supporting 'mm9', 'mm10', 'hg19', 'hg38', if using other references,
            you can simply provide the chromosome name and corresponding size (bp) with a dictionary in Python.
            e.g. {'chr1': 150000000, 'chr2': 130000000, 'chr3': 200000000}
            resolution (int): resolution: the resolution to separate genome into bins. If using .hic file format,
            the given resolution must match with the resolutions in .hic file.
            sparse (bool): whether to use sparse matrix to store (only effective when max_distance=None), default: False
            format (str): e.g., '.hic' or 'customized', default: 'customized'
            customized_format (int or list): format for each line (see README)
            chromosomes (list or str): chromosomes to use, eg. ['chr1', 'chr2'], or just 'except Y', 'except XY',
            'all', default: 'all', which means chr 1-19 + XY for mouse and chr 1-22 + XY for human
            adjust_resolution (bool): whether to adjust resolution for input file. Sometimes the input file is
            already in the proper resolution (e.g. position 3000000 has already been changed to 6 in 500kb resolution),
            then you can set adjust_resolution=False. Default: True
            keep_n_strata (None or int): only consider contacts within this genomic distance, default: None.
            If 'None', it will store full matrices in numpy matrix or scipy sparse format,
            which will use too much memory sometimes
            store_full_map (bool): whether store all contact maps
            normalization (None or str): 'OE' (observed / expected), 'VC', 'VC_SQRT', 'KR'
            operations (None or list): the methods use for pre-processing or smoothing the maps given in a list.
            The operations will happen in the given order. Operations: 'reduce_sparsity', 'convolution',
            'random_walk', 'network_enhancing'. Default: None.
            For pre-processing and smoothing operations, sometimes you need additional arguments.
            You can check docstrings for pre-processing and smoothing for more information.
        """
        self.resolution = resolution
        self.chromosomes, self.chromosome_lengths = get_chromosome_lengths(reference_genome, resolution, **kwargs)
        self.num_of_cells = len(list_of_files)
        self.sparse = sparse
        self.keep_n_strata = keep_n_strata

        res_adjust = kwargs.pop('resolution_adjust', True)
        header = kwargs.pop('header', 0)
        custom_format = kwargs.pop('customized_format', None)
        map_filter = kwargs.pop('map_filter', 0.)
        gzip = kwargs.pop('gzip', False)

        assert keep_n_strata is not None or store_full_map is True

        self.strata = {
            ch: [np.zeros((self.num_of_cells, self.chromosome_lengths[ch] - i)) for i in range(keep_n_strata)]
            for ch in self.chromosomes} if keep_n_strata else None

        if not store_full_map:
            self.full_maps = None
        elif sparse:
            self.full_maps = {ch: [None] * self.num_of_cells for ch in self.chromosomes}
        else:
            self.full_maps = {
                ch: np.zeros((self.num_of_cells, self.chromosome_lengths[ch], self.chromosome_lengths[ch]))
                for ch in self.chromosomes}

        for idx, file in enumerate(list_of_files):
            print('Processing in file: {0}'.format(file))

            for ch in self.chromosomes:
                mat, strata = load_HiC(
                    file, genome_length=self.chromosome_lengths, format=format,
                    custom_format=custom_format, header=header,
                    chromosome=ch, resolution=resolution, resolution_adjust=res_adjust,
                    map_filter=map_filter, sparse=sparse, gzip=gzip,
                    keep_n_strata=keep_n_strata, operations=operations, **kwargs)
                if store_full_map:
                    self.full_maps[ch][idx] = mat
                if keep_n_strata:
                    # self.strata[ch][idx] = strata
                    for strata_idx, stratum in enumerate(strata):
                        self.strata[ch][strata_idx][idx, :] = stratum

    def cal_strata(self, n_strata):
        if self.full_maps is None:
            if self.keep_n_strata <= n_strata:
                print(' Only {0} strata are kept!'.format(self.keep_n_strata))
                return deepcopy(self.strata)
            else:
                return deepcopy({ch: self.strata[ch][:n_strata] for ch in self.chromosomes})
        else:
            if self.keep_n_strata is None:
                new_strata = {
                    ch: [np.zeros((self.num_of_cells, self.chromosome_lengths[ch] - i))
                         for i in range(n_strata)] for ch in self.chromosomes}
                for ch in self.chromosomes:
                    for idx in range(self.num_of_cells):
                        fmap = self.full_maps[ch][idx].toarray() if self.sparse else self.full_maps[ch][idx]
                        for i in range(n_strata):
                            new_strata[ch][i][idx, :] = np.diag(fmap[i:, :-i])
                return new_strata
            elif self.keep_n_strata >= n_strata:
                return deepcopy({ch: self.strata[ch][:n_strata] for ch in self.chromosomes})
            else:
                for ch in self.chromosomes:
                    self.strata[ch] += [(np.zeros(self.num_of_cells, self.chromosome_lengths[ch] - i))
                                        for i in range(self.keep_n_strata, n_strata)]
                    for idx in range(self.num_of_cells):
                        fmap = self.full_maps[ch][idx].toarray() if self.sparse else self.full_maps[ch][idx]
                        for i in range(self.keep_n_strata, n_strata):
                            self.strata[ch][i][idx, :] = np.diag(fmap[i:, :-i])
                return deepcopy(self.strata)

    def processing(self, operations, **kwargs):
        if self.full_maps is None:
            raise ValueError('No full maps stored. Processing is not doable.')
        if self.sparse:
            for ch in self.chromosomes:
                for i, mat in enumerate(self.full_maps):
                    self.full_maps[ch][i] = coo_matrix(matrix_operation(mat.toarray(), operations, **kwargs))
        else:
            for ch in self.chromosomes:
                for i, mat in enumerate(self.full_maps):
                    self.full_maps[ch][i, :, :] = matrix_operation(mat, operations, **kwargs)

    def learn_embedding(self, similarity_method, embedding_method,
                        dim=2, aggregation='median', n_strata=None, return_distance=False, print_time=False,
                        **kwargs):
        """
        Learn a low-dimensional embedding for cells.

        Args:
            similarity_method (str): 'inner_product', 'HiCRep' or 'Selfish'
            embedding_method (str): 'MDS', 'tSNE' or 'UMAP'
            aggregation (str): 'mean' or 'median'
            dim (int): dimension of the embedding
            n_strata (int): only consider contacts within this genomic distance, default: None.
            If it is None, it will use the all strata kept (the argument keep_n_strata) from previous loading process
            return_distance (bool): if True, return (embeddings, distance_matrix); if False, only return embeddings

        Some additional argument for Selfish:
            n_windows: number of Selfish windows
            sigma: sigma in the Gaussian-like kernel

        Return:
            embeddings numpy.array: (shape: num_of_cells * dimension),
            distance_matrix numpy.array: (shape: num_of_cells * num_of_cells)
        """
        distance_matrices = []
        assert embedding_method.lower() in ['mds', 'tsne', 'umap']
        assert n_strata is not None or self.keep_n_strata is not None
        n_strata = n_strata if n_strata is not None else self.keep_n_strata
        new_strata = self.cal_strata(n_strata)
        for ch in self.chromosomes:
            print(ch)
            distance_mat = pairwise_distances(new_strata[ch], similarity_method=similarity_method,
                                              print_time=print_time, **kwargs)
            distance_matrices.append(distance_mat)
        distance_matrices = np.array(distance_matrices)

        if aggregation == 'mean':
            final_distance = np.mean(distance_matrices, axis=0)
        elif aggregation == 'median':
            final_distance = np.median(distance_matrices, axis=0)
        else:
            raise ValueError('Aggregation method {0} not supported. Only "mean" or "median".'.format(aggregation))

        embedding_method = embedding_method.lower()
        if embedding_method == 'mds':
            embeddings = MDS(final_distance, dim)
        elif embedding_method == 'tsne':
            embeddings = tSNE(final_distance, dim, **kwargs)
        else:
            embeddings = UMAP(final_distance, dim, **kwargs)

        if return_distance:
            return embeddings, final_distance
        else:
            return embeddings
