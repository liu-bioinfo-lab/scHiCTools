import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix


def processing(scHiC, methods, **kwargs):
    """
    :param methods: (str) 'reduce_sparsity', 'convolution', 'random_walk' or 'network_enhancing'

    reduce_sparsity: Hi-C reads usually varies between several magnitude,
  taking logarithm or powers might make it easy for later calculation. Arguments include:
    - sparsity_method: 'log' (log(W_ij + 1), thus 0 in original matrix is stll 0 in processed matrix)
    or 'power', default: 'log'
    - power: (if you choose sparsity_method='power') a number usually between 0 and 1.
    e.g. power=0.5 means all values W_ij in contact map will be changed to (W_ij)^0.5
    (i.e. sqrt(W_ij)), default: 0.5

    convolution: smoothing with a N by N convolution kernel, with each value equal to 1/N^2
  Argument:
    - kernel_shape: an integer. e.g. kernel_shape=3 means a 3*3 matrix with each value = 1/9, default: 3

    Random walk: multiply by a transition matrix (also calculated from contact map itself). Argument:
    - random_walk_ratio: a value between 0 and 1, e.g. if ratio=0.9, the result will be
    0.9 * matrix_after_random_walk + 0.1 * original_matrix. Default: 1.0

    Network enhancing: transition matrix only comes from k-nearest neighbors of each line.
  Arguments:
    - kNN: value 'k' in kNN, default: 20
    - iterations: number of iterations for network enhancing, default: 1
    - alpha: similar with random_walk_ratio, default: 0.9

    Return nothing. Only modify scHiC.processed_maps
    """
    for method in methods:
        for chromosome_name in scHiC.chromosomes:
            if scHiC.processed_maps[chromosome_name] is None:
                m = scHiC.maps[chromosome_name].copy()
            else:
                m = scHiC.processed_maps[chromosome_name].copy()
            if scHiC.sparse:
                m = m.toarray()

            if method == 'convolution':
                m = convolution(m, **kwargs)
            elif method == 'random_walk':
                m = random_walk(m, **kwargs)
            elif method == 'reduce_sparsity':
                m = reduce_sparsity(m, **kwargs)
            elif method == 'network_enhancing':
                m = network_enhancing(m, **kwargs)
            else:
                print('Operation not in [reduce_sparsity, convolution, random_walk, network_enhancing].\
                                 Operation omitted.')

            if scHiC.sparse:
                m = coo_matrix(m)
            scHiC.processed_maps[chromosome_name] = m


def convolution(mat, kernel_shape=3, **kwargs):
    conv = np.ones((kernel_shape, kernel_shape)) / (kernel_shape ** 2)
    mat = convolve2d(mat, conv, 'same')
    return mat


def random_walk(mat, random_walk_ratio=1.0, **kwargs):
    sm = np.sum(mat, axis=1)
    sm = np.where(sm == 0, 1, sm)
    sm = np.tile(sm, (len(mat), 1)).T
    walk = mat / sm
    mat = random_walk_ratio * walk.T.dot(mat).dot(walk) + (1 - random_walk_ratio) * mat
    return mat


def reduce_sparsity(mat, sparsity_method='log', power=0.5, **kwargs):
    if sparsity_method == 'log':
        return np.log(mat + 1)
    elif sparsity_method == 'power':
        return np.power(mat, power)
    else:
        raise ValueError('Method {0} not supported while reducing sparsity.'.format(sparsity_method))


def network_enhancing(mat, kNN=20, iteration=1, alpha=0.9, **kwargs):
    argsort = np.argsort(-mat, axis=1)
    new_mat = np.array(mat.shape)
    for i in range(len(mat)):
        for j in range(kNN):
            pos = argsort[i, j]
            new_mat[i, pos] = mat[i, pos]

    sm = np.sum(new_mat, axis=1)
    sm = np.where(sm == 0, 1, sm)
    sm = np.tile(sm, (len(mat), 1)).T
    walk = new_mat / sm

    for k in range(iteration):
        mat = alpha * walk.T.dot(mat).dot(walk) + (1 - alpha) * mat
    return mat
