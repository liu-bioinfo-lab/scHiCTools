import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import cosine


def MDS(mat, n=2):
    h = np.eye(len(mat)) - np.ones(mat.shape) / len(mat)
    k = -0.5 * h.dot(mat * mat).dot(h)
    w, v = np.linalg.eig(k)
    max_ = np.argsort(w)[:-n - 1:-1]
    co = np.real(v[:, max_].dot(np.sqrt(np.diag(w[max_]))))
    # co = np.real(v[:, :2].dot(np.sqrt(np.diag(w[:2]))))
    return co


def pairwise_distances(all_stripes, method, **kwargs):
    if method == 'inner_product':
        print(' Calculating z-scores...')
        zscores = []
        for stripes in all_stripes:
            zs = []
            for stripe in stripes:
                z = zscore(stripe) if np.sum(stripe) != 0 else stripe
                zs += z.tolist()
            zscores.append(zs)
        count, total = 0, (len(all_stripes) - 1) * (len(all_stripes) - 2) // 2
        distance_mat = np.zeros((len(zscores), len(zscores)))
        for i in range(len(zscores) - 1):
            for j in range(i+1, len(zscores)):
                if count % 10000 == 0:
                    print(' Distances: {0} / {1}'.format(count, total))
                count += 1
                distance = np.sqrt(2 - 2 * np.inner(zscores[i], zscores[j]))
                distance_mat[i][j] = distance
                distance_mat[j][i] = distance

    elif method == 'HiCRep':
        print(' Calculating means and stds...')
        all_means = np.zeros((len(all_stripes), len(all_stripes[0])))
        all_stds = np.zeros((len(all_stripes), len(all_stripes[0])))
        for i in range(len(all_stripes)):
            for j in range(len(all_stripes[i])):
                all_means[i, j] = np.mean(all_stripes[i][j])
                if all_means[i, j] != 0:
                    all_stds[i, j] = np.std(all_stripes[i][j])
        count, total = 0, (len(all_stripes) - 1) * (len(all_stripes) - 2) // 2
        distance_mat = np.zeros((len(all_stripes), len(all_stripes)))
        for i in range(len(all_stripes) - 1):
            for j in range(i+1, len(all_stripes)):
                if count % 10000 == 0:
                    print(' Distances: {0} / {1}'.format(count, total))
                count += 1
                numerator = sum([np.inner(stripe_1 - all_means[i, ii], stripe_2 - all_means[j, ii])
                                 for ii, (stripe_1, stripe_2) in enumerate(zip(all_stripes[i], all_stripes[j]))])
                denominator = sum([len(all_stripes[i][k]) * all_stds[i][k] * all_stds[j][k]
                                   for k in range(len(all_stripes[i]))])
                distance = np.sqrt(2 - 2 * numerator / denominator)
                distance_mat[i][j] = distance
                distance_mat[j][i] = distance

    elif method == 'Selfish':
        pass

    else:
        raise ValueError('Method {0} not supported. Only "inner_product", "HiCRep" and "Selfish".'.format(method))







