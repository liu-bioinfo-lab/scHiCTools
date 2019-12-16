import numpy as np
from .embedding_utils import tsne_search_prob


def MDS(mat, n=2):
    # mat = np.sqrt(2 - 2 * mat)
    h = np.eye(len(mat)) - np.ones(mat.shape) / len(mat)
    k = -0.5 * h.dot(mat * mat).dot(h)
    w, v = np.linalg.eig(k)
    max_ = np.argsort(w)[:-n - 1:-1]
    co = np.real(v[:, max_].dot(np.sqrt(np.diag(w[max_]))))
    # co = np.real(v[:, :2].dot(np.sqrt(np.diag(w[:2]))))
    return co


def tSNE(mat, dim=2, **kwargs):
    """
    Runs t-SNE on N * N distance matrix to reduce its dimensionality.
    Referred to sklearn.manifold.t_sne and
    https://blog.csdn.net/Flyingzhan/article/details/79521765

    Args:
        mat (numpy.array): N * N distance matrix
        dim (int): dimension of embedded vectors

    Additional Args:
        tSNE_n_iter (int): iteration steps

    Return:
         co (numpy.array): embedded coordinates
    """
    n_samples = mat.shape[0]
    assert isinstance(dim, int)

    n_iter = kwargs.pop('tSNE_n_iter', 500)
    initial_momentum = kwargs.pop('tSNE_initial_momentum', 0.5)
    final_momentum = kwargs.pop('tSNE_final_momentum', 0.8)
    eta = 500
    min_gain = 0.01

    y = np.random.randn(n_samples, dim)
    dy = np.zeros((n_samples, dim))
    iy = np.zeros((n_samples, dim))
    gains = np.ones((n_samples, dim))

    P = tsne_search_prob(mat, tol=kwargs.pop('tSNE_tol', 1e-5),
                         perplexity=kwargs.pop('tSNE_perplexity', 30),
                         initial_sigma=kwargs.pop('tSNE_initial_sigma', 1.0))

    P = np.maximum(4 * P, 1e-12)

    # @Xinjun, 以下这部分我还没有检查梯度下降写得对不对，
    # 请参考https://blog.csdn.net/Flyingzhan/article/details/79521765

    for iter in range(n_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        for i in range(n_samples):
            num[i, i] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        momentum = initial_momentum if iter <= n_iter // 10 else final_momentum

        # Compute gradient
        delta = P - Q
        for i in range(n_samples):
            dy[i, :] = np.sum(np.tile(delta[:, i] * num[:, i], (dim, 1)).T * (y[i, :] - y), axis=0)

        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n_samples, 1))
        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum(P / 4 * np.log(P / 4 / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
        # Stop lying about P-values
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y


def UMAP(mat, n=2, **kwargs):
    pass

