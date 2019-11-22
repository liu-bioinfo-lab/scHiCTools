import numpy as np


def tsne_cal_perplexity(dist, idx=0, beta=1.0):
    '''
    Calculating perplexity for one sample (one data point)

    Args:
        dist: corresponding line in distance matrix
        idx: id of the sample
        beta: beta = 1 / (2 * sigma^2) in Gaussian distribution

    Return:
        perp: perplexity
        prob: probabilities from Gaussian distribution
    '''
    # p_ij = exp(-beta*d_ij) / (sum_j(exp(-beta*d_ij)))
    prob = np.exp(-dist * beta)
    prob[idx] = 0  # p_ii = 0
    sum_prob = np.sum(prob)

    # log(perp) = - sum_j(p_ij log(p_ij))
    # log(perp) = - sum_j{exp(-beta*d_ij)/(sum_j(exp(-beta*d_ij))) * log[exp(-beta*d_ij)/(sum_j(exp(-beta*d_ij)))]}
    # log(perp) = - sum_j{exp(-beta*d_ij)/(sum_j(exp(-beta*d_ij))) * log[exp(-beta*d_ij)]} +
    #   sum_j{exp(-beta*d_ij)/(sum_j(exp(-beta*d_ij))) * log[(sum_j(exp(-beta*d_ij)))]}
    # log(perp) = sum_j(beta*exp(-beta*d_ij)*d_ij)/sum_j(exp(-beta*d_ij)) + log(sum_j(exp(-beta*d_ij)))
    perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob

    prob /= sum_prob
    return perp, prob


def tsne_search_prob(dist, tol=1e-5, perplexity=30.0, initial_sigma=1.0):
    '''
    Obtain a best beta value for each data point and calculate pairwise probabilities

    Args:
        dist: distance matrix
        tol: tolerance
        perplexity: initial perplexity
        initial_sigma: beta = 1 / (2 * sigma^2)

    Return:
        pair_prob: pairwise probability matrix
    '''

    # 初始化参数
    print("Computing pairwise distances...")
    pair_prob = np.zeros(dist.shape)
    n_samples = dist.shape[0]
    beta = np.ones((n_samples, 1)) / 2 / (initial_sigma ** 2)  # initial value of beta
    base_perp = np.log(perplexity)  # use logarithm

    for i in range(n_samples):
        # if i % 100 == 0:
        #   print("Computing pair_prob for point %s of %s ..." % (i, n_samples))

        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i, :], i, beta[i])

        # binary search
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 100:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # update perb and prob
            perp, this_prob = tsne_cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        pair_prob[i, ] = this_prob
    # print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    pair_prob = pair_prob + pair_prob.T
    pair_prob = pair_prob / np.sum(pair_prob)  # symmetric matrix
    return pair_prob

