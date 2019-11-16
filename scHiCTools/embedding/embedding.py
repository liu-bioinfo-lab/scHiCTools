import numpy as np


def MDS(mat, n=2):
    h = np.eye(len(mat)) - np.ones(mat.shape) / len(mat)
    k = -0.5 * h.dot(mat * mat).dot(h)
    w, v = np.linalg.eig(k)
    max_ = np.argsort(w)[:-n - 1:-1]
    co = np.real(v[:, max_].dot(np.sqrt(np.diag(w[max_]))))
    # co = np.real(v[:, :2].dot(np.sqrt(np.diag(w[:2]))))
    return co


def tSNE(mat, n=2, **kwargs):
    pass


def UMAP(mat, n=2, **kwargs):
    pass

