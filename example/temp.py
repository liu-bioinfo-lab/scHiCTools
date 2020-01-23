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
x = scHiCs(['cell_01', 'cell_02', 'cell_03'], reference_genome='mm9', resolution=500000, keep_n_strata=10,
           format='shortest_score', resolution_adjust=False, chromosomes=['chr1'], store_full_map=True
           )
print(x.learn_embedding('inner_product', 'MDS', dim=2, n_strata=10))

x.processing(['random_walk', 'network_enhancing'])
print(x.learn_embedding('inner_product', 'MDS', dim=2, n_strata=10))
