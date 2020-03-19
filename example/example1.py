
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

print(sys.path)
from scHiCTools import scHiCs

x = scHiCs(['cell_01','cell_02','cell_03'], reference_genome='mm9', resolution=500000, max_distance=4000000,
           format='shortest_score', resolution_adjust=True, chromosomes='except Y',
           operations=['convolution'], kernel_shape=3, keep_n_strata=10, store_full_map=False
           )

x = scHiCs(, reference_genome='hg38', resolution=500000, max_distance=4000000,
           format='mcool', resolution_adjust=False, chromosomes='except Y',
           operations=['convolution'], kernel_shape=3, keep_n_strata=10, store_full_map=False
           )

emb, _ = x.learn_embedding(similarity_method='innerproduct', embedding_method='MDS', aggregation='median', print_time=False, return_distance=True)

print(emb)
print(_)

