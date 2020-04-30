import sys
import os
sys.path.insert(0, './')
import numpy as np
from scHiCTools import scHiCs
from scHiCTools import scatter
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


files=os.listdir(os.getcwd()+'/1CDX_cells')
for i in range(len(files)):
    files[i]='1CDX_cells/'+files[i]+'/new_adj'
# files=np.random.choice(files, size=20, replace=False)


x = scHiCs(files,
           reference_genome='mm9',
           resolution=500000,
           max_distance=4000000,
           format='shortest_score',
           resolution_adjust=True,
           chromosomes='except Y',
           # operations=['convolution'],
           kernel_shape=3,
           keep_n_strata=10,
           # store_full_map=True
           )

x.plot_contacts()


x.select_cells(min_n_contacts=300000,max_short_range_contact=.9)


x.plot_contacts()

plt.figure()
plt.hist(x.contacts)
plt.xlabel("Number of contacts")
plt.ylabel('Frequency')
plt.title('Histogram of contacts')
plt.show()


emb=x.learn_embedding('inner_product', 'MDS', dim=2, n_strata=10)
hicluster=x.scHiCluster()
label=x.clustering(n_clusters=4, clustering_method='kmeans', similarity_method='innerproduct', aggregation='median')

print(emb)
print(hicluster)
print(label)


fig=plt.figure()
plt.subplot(121)

ax1=plt.subplot(1, 2, 1)
scatter(emb,label=label,point_size=5)
ax1.set_title('InnerProduct + MDS + Kmeans embedding')

# add a subplot with no frame
ax2=plt.subplot(1, 2, 2)
scatter(hicluster[0],label=hicluster[1],point_size=5)
ax2.set_title('scHiCluster')

plt.show()

