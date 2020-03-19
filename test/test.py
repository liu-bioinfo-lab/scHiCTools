

import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scHiCTools import scHiCs
from scHiCTools import scatter

files=os.listdir('../example/1CDX_cells')
for i in range(len(files)):
    files[i]='../example/1CDX_cells/'+files[i]+'/new_adj'



# x = scHiCs(files, reference_genome='mm9', resolution=500000, max_distance=4000000,
#            format='shortest_score', resolution_adjust=True, chromosomes='except Y',
#            operations=['convolution'], kernel_shape=3, keep_n_strata=10, store_full_map=False
#            )

y = scHiCs(files, reference_genome='mm9', resolution=500000, max_distance=4000000,
           format='shortest_score', resolution_adjust=True, chromosomes='ecept Y',
           operations=['convolution'], kernel_shape=3, keep_n_strata=10, store_full_map=True
           )


emb1 = y.learn_embedding(similarity_method='innerproduct',return_distance=True, embedding_method='mds', aggregation='median', print_time=False)


emb2=y.learn_embedding(similarity_method='HiCRep',return_distance=True, embedding_method='mds', aggregation='median', print_time=False)

emb3 = y.learn_embedding(similarity_method='Selfish',return_distance=True, embedding_method='mds', aggregation='median', print_time=False)

emb4 = y.learn_embedding(similarity_method='innerproduct',return_distance=True, embedding_method='mds', aggregation='mean', print_time=False)

emb5 = y.learn_embedding(similarity_method='innerproduct',return_distance=True, embedding_method='tSNE', aggregation='median', print_time=False)

emb6 = y.learn_embedding(similarity_method='innerproduct',return_distance=True, embedding_method='UMAP', aggregation='median', print_time=False)

emb7 = y.learn_embedding(similarity_method='innerproduct',return_distance=True, embedding_method= 'phate', aggregation='median', print_time=False)

emb8 = y.learn_embedding(similarity_method='innerproduct',return_distance=True, embedding_method= 'spectral_embedding', aggregation='median', print_time=False)

label1=y.clustering(n_clusters=4, clustering_method='kmeans', similarity_method='innerproduct', aggregation='median', n_strata=None)

label2=y.clustering(n_clusters=4, clustering_method='spectral_clustering', similarity_method='innerproduct', aggregation='median', n_strata=None)

hicluster=y.scHiCluster(dim=2,cutoff=0.8,n_PCs=10,n_clusters=4)


import matplotlib.pyplot as plt

plt.figure()
plt.subplot(1,2,1)
scatter(emb1[0]*100, label=label1)
plt.subplot(1,2,2)
scatter(emb2[0])
plt.show()

plt.figure()
scatter(hicluster[0],label=hicluster[1])
plt.show()

print(emb1)
