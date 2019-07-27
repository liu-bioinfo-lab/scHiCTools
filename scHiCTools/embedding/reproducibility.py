import numpy as np
from scipy.stats import zscore
# from scipy.spatial.distance import cosine


def pairwise_distances(all_stripes, method, n_windows=10, sigma=1.6):
    if method == 'InnerProduct':
        print(' Calculating z-scores...')
        zscores = []
        for stripes in all_stripes:
            zs = []
            for stripe in stripes:
                z = zscore(stripe) if np.sum(stripe) != 0 else stripe
                zs += z.tolist()
            zscores.append(zs)
        count, total = 0, (len(all_stripes) - 1) * len(all_stripes) // 2
        distance_mat = np.zeros((len(zscores), len(zscores)))
        for i in range(len(zscores) - 1):
            for j in range(i+1, len(zscores)):
                if count % 10000 == 0:
                    print(' Distances: {0} / {1}'.format(count, total))
                count += 1
                distance = np.sqrt(2 - 2 * np.inner(zscores[i], zscores[j]) / len(zscores[i]))
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

    elif method == 'Selfish':
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
        for idx, fingerprint in fingerprints:
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

