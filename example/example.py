import sys
sys.path.insert(0, '../')
print(sys.path)
from scHiCTools import scHiCs


x = scHiCs(['cell_01', 'cell_02', 'cell_03'], reference_genome='mm9', resolution=500000, max_distance=4000000,
           format='customized', adjust_resolution=False, line_format=12345, header=False, chromosomes='no Y',
           preprocessing=['reduce_sparsity', 'convolution'], kernel_shape=3
           )
emb = x.learn_embedding(method='InnerProduct', aggregation='median')

print(emb)
