import sys
sys.path.insert(0, '../')
print(sys.path)
from scHiCTools import scHiCs


x = scHiCs(['cell_01', 'cell_02', 'cell_03'], reference_genome='mm9', resolution=500000, max_distance=4000000,
           format='shortest_score', adjust_resolution=False, chromosomes='no Y',
           operations=['reduce_sparsity', 'convolution'], kernel_shape=3, keep_n_strata=10, store_full_map=False
           )
emb = x.learn_embedding(similarity_method='InnerProduct', embedding_method='MDS', aggregation='median')

print(emb)
