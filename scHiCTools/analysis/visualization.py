# -*- coding: utf-8 -*-
"""

 Visualization component of scHiCTools

 Author: Xinjun Li

 This script define a function to plot scatter plot of embedding points
     of single cell data.

"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter(data, dimension="2D", point_size=3, sty='default',
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


    mpl.style.use(sty)

    # 2D scatter plot
    if dimension=="2D":

        # Plot with label
        if label is not None:
            lab=list(set(label))
            for index, l in enumerate(lab):
                plt.scatter(data[label==l,0], data[label==l,1],
                                c='C{!r}'.format(index),
                                s=point_size, label=l,
                                alpha=alpha, **kwargs)
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

            for index, l in enumerate(lab):
                splot.scatter(data[label==l,0], data[label==l,1], data[label==l,2],
                              s=point_size,
                              color='C{!r}'.format(index),
                              label=l)
            plt.legend(**kwargs)
        # Plot without label
        else:
            splot.scatter(data[:,0], data[:,1], data[:,2],s=point_size)

        if aes_label is not None:
            splot.set_xlabel(aes_label[0])
            splot.set_ylabel(aes_label[1])
            splot.set_zlabel(aes_label[2])

    plt.title(title)

