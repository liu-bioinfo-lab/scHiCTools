import numpy as np


def normalization_matrix(scHiC, method):
    if method not in ['OE', 'KR', 'VC', 'VC_SQRT']:
        print("Normalization operation not in ['OE', 'KR', 'VC', 'VC_SQRT']. Normalization omitted.")

    return np.array([0])

