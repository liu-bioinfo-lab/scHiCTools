"""

 Visualization component of scHiCTools
 
 Author: Xinjun Li
 
 This script define a function to plot scatter plot of embedding points
     of single cell data.

"""


import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter(data, dimension="2D", point_size=3, cmap='Sequential',
            label=None, title=None, alpha=None, aes_label=None,
            **kwargs):
    """
    This function is to plot scatter plot of embedding points
        of single cell data.
    
    inputs:
        data: a numpy array which has 2 or 3 columns, every row represent a point.
        dimension: specifiy the dimension of the plot, either "2D" or "3D".
        label: specifiy the label of each point.
        
    outputs:
        Scatter plot of either 2D or 3D.
    
    """
    
    # Error messages.
    if dimension not in ["2D", "3D"]:
        raise ValueError('Dimension must be "2D" or "3D".')
    if (dimension=="2D" and len(data[0])!=2) or (dimension=="3D" and len(data[0])!=3):
        raise ValueError('Data shape must match dimension!')
    if (label is not None) and len(data)!=len(label):
        raise ValueError('Number of rows in data must equal to length of label!')
    
    
    # cmaps need revise ?
    cmaps = {
            'Perceptually Uniform Sequential':
                ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            'Sequential':
                ['b', 'r', 'c', 'y', 'm'],
#                ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
            'Sequential (2)':
                ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'],
            'Diverging': 
                ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'],
            'Cyclic': 
                ['twilight', 'twilight_shifted', 'hsv'],
            'Qualitative': 
                ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'],
            'Miscellaneous': 
                ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']}
    
    
    
    # 2D scatter plot
    if dimension=="2D":
        
        # Plot with label
        if label is not None:
            fig, subplot = plt.subplots()
            lab=list(set(label))
            color=cmaps[cmap][:len(lab)]
            for index, l in enumerate(lab):
                subplot.scatter(data[label==l,0], data[label==l,1],
                                c=color[index], s=point_size, label=l,
                                alpha=alpha, edgecolors='none')
            plt.legend(**kwargs)
        # Plot without label
        else:
            plt.scatter(data[:,0], data[:,1], s=point_size, alpha=alpha)
        
        if aes_label is not None:
                plt.xlabel(aes_label[0])
                plt.ylabel(aes_label[1])
        
        
    # 3D scatter plot
    if dimension=="3D":
        splot = plt.subplot(111, projection='3d')
        
        # Plot with label
        if label is not None:
            lab=list(set(label))
            
            color=cmaps[cmap][:len(lab)]
             
            for index, l in enumerate(lab):
                splot.scatter(data[label==l,0], data[label==l,1], data[label==l,2], s=point_size,
                              color=color[index], label=l)
            plt.legend(**kwargs)
        # Plot without label
        else:
            splot.scatter(data[:,0], data[:,1], data[:,2],s=point_size)
            
        if aes_label is not None:
            splot.set_xlabel(aes_label[0])
            splot.set_ylabel(aes_label[1])
            splot.set_zlabel(aes_label[2])
    
    plt.title(title)
    plt.show()



