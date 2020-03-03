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
x = scHiCs(['4DNFIW961HN2.mcool'], reference_genome='hg38', resolution=500000, max_distance=4000000,
           format='mcool', resolution_adjust=False, chromosomes='except Y',
           operations=['convolution'], kernel_shape=3, keep_n_strata=10, store_full_map=False
           )
# print('Time 0:', time() - t0)
# emb, _ = x.learn_embedding(similarity_method='selfish', embedding_method='MDS', aggregation='median', print_time=False, return_distance=True)
# print(emb)
# print(_)

print(x)


