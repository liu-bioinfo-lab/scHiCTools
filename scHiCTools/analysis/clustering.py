"""

 Clustering component of scHiCTools

 Author: Xinjun Li

 This script define functions to clustering single cell data.

"""

import numpy as np
from scipy.sparse import csgraph


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




# spectral_clustering can directly takes similarity matrix.
def spectral_clustering(data,
                        data_type='graph',
                        n_clusters=4,
                        method='Unnormalized'):

    """
    This function is a impliment of unnormalized spectral clustering.

    Input:
        data: a numpy array, can either be a matrix which every row represent a point
            or a graph similarity matrix.
        data_type: specify the type of data input.
        k: number of clusters.

    Output:
        a list of the label of the data points.
    """    

    # compute graph similarity matrix
    if method=='Unnormalized':
        if data_type=='graph' and len(data)==len(data[0]):
            if len(data)!=len(data[0]):
                raise ValueError('data is not a distance matrix!')
            elif np.sum(data.T!=data)>0:
                raise ValueError('data is not a distance matrix!')
            elif sum(np.diag(data)!=0)!=0:
                raise ValueError('data is not a distance matrix!')
            graph=data
        elif data_type=='points':
            # Use pairwised Euclidean distances to compute graph
            graph = np.sum(np.square(data), 1)
            graph = np.add(np.add(-2 * np.dot(data, data.T), graph).T, graph)
        else:
            raise ValueError('data_type is not supported.')

        # the unnormalized Laplacian L.
        L=csgraph.laplacian(graph, normed=False)

        # eigenvalues and eigenvectors.
        eig_vals, eig_vecs = np.linalg.eig(L)
        eig_vecs= eig_vecs[:, np.flipud(eig_vals.argsort())]
        eig_vals= np.flipud(np.sort(eig_vals))

        # first k=n_clusters eigenvectors.
        U=eig_vecs[:,range(n_clusters)]

        # Clustering using k-means
        label=kmeans(U,k=n_clusters)

        return(label)
