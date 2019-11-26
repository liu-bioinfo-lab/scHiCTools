"""

 Clustering component of scHiCTools

 Author: Xinjun Li

 This script define functions to clustering single cell data.

"""

import numpy as np


def kmeans(data, k=4,
           weights=None, iteration=1000):
    """
    This function is a impliment of K-Means algorithm.

    Input:
        data: a numpy array, every row represent a point.
        k: number of clusters.
        weights: list of the weight of every point, should either be 'None' or
                have the length same as the number of rows in data.

    Output:
        a list of the label of every point.
    """

    # Random initialize centroids
    centroids=np.array(range(len(data)))
    np.random.shuffle(centroids)
    centroids=data[centroids[:k],:]
    label=np.zeros(len(data))

    # loop to find the proper cluster
    for i in range(iteration):

        # calculate the distance to the centroids respectively
        dist=np.zeros((len(data), k))
        for cent in range(k):
            dist[:,cent]=np.sum(np.square(data-centroids[cent]), axis=1)

        # find the lable under the centroids
        label0=np.argmin(dist,axis=1)
        if sum(label!=label0)==0:
            break
        else:
            label=label0

        # renew centroids
        if weights is not None:
            for j in range(k):
                group=data[label==j,:]
                centroids[j]=np.average(group, axis=0,weights=weights[label==j])
        else:
            for j in range(k):
                group=data[label==j,:]
                centroids[j]=np.average(group, axis=0)

    return(label)
