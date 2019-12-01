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






# Two types of hierarchical clustering: top-down & bottom-up

# Bottom-up algorithms: hierarchical agglomerative clustering(HAC)
# HAC can directly take distance matrix.
def HAC(data,
        data_type='points',
        n_clusters=4,
        method="single-link"):

    """
    This function is a impliment of hierarchical afflomerative clustering.
    Reference: Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze,
            "Introduction to Information Retrieval" chapter 17

    Input:
        data: a numpy array,
            can either be a matrix which every row represent a point
            or a graph similarity matrix.
        n_clusters: number of clusters.
        method: now support "single-link", "complete-link".

    Output:
        a list of the label of the data points.

    """

    if method=='single-link':
        # compute graph similarity matrix
        if data_type=='graph' and len(data)==len(data[0]):
            if len(data)!=len(data[0]):
                raise ValueError('data is not a distance matrix!')
            elif np.sum(data.T!=data)>0:
                raise ValueError('data is not a distance matrix!')
            elif sum(np.diag(data)!=0)!=0:
                raise ValueError('data is not a distance matrix!')
            # using fully connected graph with Gaussian similarity function
            graph=np.exp(-np.square(data)/np.mean(data)) # sigma?
        elif data_type=='points':
            # pair-wise Euclidean distance
            graph = np.sum(np.square(data), 1)
            graph = np.add(np.add(-2 * np.dot(data, data.T), graph).T, graph)
            # using fully connected graph with Gaussian similarity function
            graph = np.exp(-graph/np.mean(graph)) # sigma?
        else:
            raise ValueError('data_type is not supported.')

        similarity=graph-np.diag(np.diag(graph))
        label=np.array(range(len(graph)))

        while max(label)>=n_clusters:

            # find maximum similarity between groups
            index=np.argmax(similarity)
            index=[index//len(similarity), index%len(similarity)]

            # combine similarity groups
            label[label==index[1]]=index[0]
            label[label>index[1]]-=1

            # new similarity matrix
            new_simi=np.amax(similarity[:,index],axis=1)
            similarity[:,index[0]]=new_simi
            similarity[index[0],:]=new_simi
            similarity=np.delete(
                    np.delete(similarity,index[1],axis=0),
                    index[1],axis=1)
            similarity[index[0],index[0]]=0


    return(label)
