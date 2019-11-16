import numpy as np
from scipy.stats import zscore


def pairwise_distances(all_strata, similarity_method, **kwargs):
    """
    Args:
        all_strata (numpy.array): [n_cells * n_bins, n_cells * (n_bins - 1), ...]
        similarity_method (str): 'inner_product', 'HiCRep', 'Selfish'
        n_windows (int):
        sigma (float):

    Return:
         distance_mat (numpy.array)
    """
    method = similarity_method.lower()

    if method in ['inner_product', 'innerproduct']:
        print(' Calculating z-scores...')
        zscores = []
        for stratum in all_strata:
            zscores.append(zscore(stratum, axis=1))
        zscores = np.concatenate(zscores, axis=1)

        print(' Calculating inner product...')
        inner = zscores.dot(zscores.T) / zscores.shape[1]
        distance_mat = np.sqrt(2 - 2 * inner)

    elif method == 'hicrep':
        print(' Calculating means and stds...')
        all_means = np.zeros((len(all_stripes), len(all_stripes[0])))
        all_stds = np.zeros((len(all_stripes), len(all_stripes[0])))
        for i in range(len(all_stripes)):
            for j in range(len(all_stripes[i])):
                all_means[i, j] = np.mean(all_stripes[i][j])
                if all_means[i, j] != 0:
                    all_stds[i, j] = np.std(all_stripes[i][j])
        count, total = 0, (len(all_stripes) - 1) * len(all_stripes) // 2
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

    elif method == 'selfish':
        map_size = len(all_stripes[0][0])  # length of the contact map for this chromosome
        step_size = map_size // (2 * n_windows)
        fingerprints = [np.zeros(2 * n_windows - 1,) for i in range(len(all_stripes))]
        print(' Calculating summation of sliding windows...')
        for idx, stripes in enumerate(all_stripes):
            for i in range(2 * n_windows - 1):
                start_pos, end_pos = step_size * i, step_size * (i + 2)
                sm = 0
                for j, stripe in enumerate(stripes):
                    if j >= 2 * step_size:
                        continue
                    sm += np.sum(stripe[start_pos: end_pos - j])
                fingerprints[idx][i] = sm
        print(' Pairwisely compare the windows...')
        for idx, fingerprint in enumerate(fingerprints):
            comp = np.zeros((len(fingerprint), len(fingerprint)))
            for i in range(len(fingerprint)-1):
                for j in range(i+1, len(fingerprint)):
                    if fingerprint[i] > fingerprint[j]:
                        comp[i, j] = 1
                    elif fingerprint[i] < fingerprint[j]:
                        comp[j, i] = 1
            fingerprints[idx] = comp

        count, total = 0, (len(all_stripes) - 1) * len(all_stripes) // 2
        distance_mat = np.zeros((len(all_stripes), len(all_stripes)))
        for i in range(len(all_stripes) - 1):
            for j in range(i + 1, len(all_stripes)):
                if count % 10000 == 0:
                    print(' Distances: {0} / {1}'.format(count, total))
                f1, f2 = fingerprints[i], fingerprints[j]
                dis = np.sum(np.square(f1 - f2))
                distance = np.sqrt(2 - 2 * np.exp(-sigma * dis))
                distance_mat[i][j] = distance
                distance_mat[j][i] = distance

    else:
        raise ValueError('Method {0} not supported. Only "inner_product", "HiCRep" and "Selfish".'.format(method))

    return distance_mat

