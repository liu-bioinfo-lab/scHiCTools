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
x = scHiCs(['cell_01'], reference_genome='mm9', resolution=500000, max_distance=4000000,
           format='shortest_score', resolution_adjust=False, chromosomes=['chr1'], store_full_map=True
           )
print(x.full_maps['chr1'][0])

x.processing(['random_walk', 'network_enhancing'])
print(x.full_maps['chr1'][0])

x = scHiCs(['cell_01'], reference_genome='mm9', resolution=500000, max_distance=4000000,
           format='shortest_score', resolution_adjust=False, chromosomes=['chr1'], store_full_map=True,
           operations=['random_walk', 'network_enhancing'])
print(x.full_maps['chr1'][0])




