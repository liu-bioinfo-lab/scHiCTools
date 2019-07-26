# scHiCTools

### Summary
A computational toolbox for analyzing single cell Hi-C (high-throughput sequencing for 3C) data which includes functions for:
1. Load single-cell HiC datasets
2. Smoothing the contact maps with linear convolution, random walk or network enhancing
3. Calculating embeddings for single cell HiC datasets efficiently with reproducibility measures include InnerProduct, fastHiCRep and Selfish

### Installation
  **Required Python Packages**
  - Python (version >= 3.6)
  - numpy (version >= 1.15.4)
  - scipy (version >= 1.0)
  
  **Install from GitHub**
  
### Usage
  **Import Package**
  ```console
  >>>import scHiCTools
  ```
  
  **Load scHiC data**
  
  The scHiC data is stored in a series of files, with each of the files corresponding to one cell.
  You need to specify the list of scHiC file paths.
  ```console
  >>>from scHiCTools import scHiCs
  >>>files = ['./cell_1', './cell_2', './cell_3']
  >>>loaded_data = scHiCs(
  ... files, reference_genome='mm9',
  ... resolution=500000, max_distance=4000000,
  ... format='customized', adjust_resolution=False,
  ... line_format=12345, header=False, chromosomes='no Y',
  ... preprocessing=['reduce_sparsity', 'smooth'],
  ... smoothing_parameter=2
  ... )
  ```
  - reference genome: now supporting 'mm9', 'mm10', 'hg19', 'hg38',
  if using other references, you can simply provide the chromosome name and
  corresponding size (bp) with a dictionary in Python.
  - resolution: the resolution to separate genome into bins.
  If using .hic file format, the given resolution must match with the
  resolutions in .hic file.
  - max_distance: only consider contacts within this genomic distance
  - format: file format, '.hic' or 'customized', if 'customized', you need to
  provide format for each line
  - header: whether the files have a header line
  - line_format:
  - chromosomes: chromosomes to use, eg. ['chr1', 'chr2'], or
  just 'except Y', 'except XY'
  - preprocessing: the methods use for pre-processing or smoothing the maps
  given in a list. The operations will happen in the given order.
  Operations: 'reduce_sparsity', 'convolution', 'random_walk', 'network_enhancing'
  - For operations, sometimes you need additional arguments (introduced in next sub-section)

  **Pre-processing and Smoothing**
  
  **Learn Embeddings**
  ```console
  >>>embs = loaded_data.learn embedding(dim=2, method='inner_product', aggregation='median')
  ```
  This function will return the embeddings in the format of a numpy array with shape (#_of_cells, #_of_dimensions).
  - dim: the dimension for the embedding
  - method: reproducibility measure, 'InnerProduct', 'fastHiCRep' or 'Selfish'
  - aggregation: method to aggregate different chromosomes, 'mean' or 'median'
  
### Citation


### References
A. R. Ardakany, F. Ay, and S. Lonardi.  Selfish: Discovery of differential chromatininteractions via a self-similarity measure.bioRxiv, 2019.

N. C. Durand, J. T. Robinson, M. S. Shamim, I. Machol, J. P. Mesirov, E. S. Lander, and E. Lieberman Aiden. "Juicebox provides a visualization system for Hi-C contact maps with unlimited zoom." Cell Systems 3(1), 2016.

J. Liu, D. Lin, G. Yardimci, and W. S. Noble. Unsupervised embedding of single-cellHi-C data.Bioinformatics, 34:96–104, 2018.

T. Nagano, Y. Lubling, C. Várnai, C. Dudley, W. Leung, Y. Baran, N. M. Cohen,S.  Wingett,  P.  Fraser,  and  A.  Tanay.    Cell-cycle  dynamics  of  chromosomal organization at single-cell resolution.Nature, 547:61–67, 2017.

B. Wang, A. Pourshafeie, M. Zitnik, J. Zhu, C. D. Bustamante, S. Batzoglou, andJ.  Leskovec.   Network  enhancement  as  a  general  method  to  denoise  weighted biological networks.Nature Communications, 9(1):3108, 2018.

T. Yang, F. Zhang, G. G. Y. mcı, F. Song, R. C. Hardison, W. S. Noble, F. Yue, andQ. Li. HiCRep: assessing the reproducibility of Hi-C data using a stratum-adjusted correlation coefficient.Genome Research, 27(11):1939–1949, 2017.

G.  G.  Yardımcı,   H.  Ozadam,   M.  E.  Sauria,   O.  Ursu,   K.-K.  Yan,   T.  Yang,A. Chakraborty, A. Kaul, B. R. Lajoie, F. Song, et al. Measuring the reproducibilityand quality of hi-c data.Genome Biology, 20(1):57, 2019.

