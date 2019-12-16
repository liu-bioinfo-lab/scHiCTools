from time import time
import sys
sys.path.insert(0, '../')
print(sys.path)
from scHiCTools import scHiCs


# Learn the embeddings of cell_01, cell_02 and cell_03
# Time:
# Time 0: input I/O and pre-processing
# Time 1: process each strata
# Time 2: calculating pairwise similarity

t0 = time()
x = scHiCs(['cell_01', 'cell_02', 'cell_03'], reference_genome='mm9', resolution=500000, max_distance=4000000,
           format='shortest_score', resolution_adjust=False, chromosomes='except Y',
           operations=['convolution'], kernel_shape=3, keep_n_strata=10, store_full_map=False
           )
print('Time 0:', time() - t0)
emb = x.learn_embedding(similarity_method='old_hicrep', embedding_method='MDS', aggregation='median', print_time=True)
print(emb)




