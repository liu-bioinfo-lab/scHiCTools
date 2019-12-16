import numpy as np
from scipy.stats import zscore


# x = np.array([[1, 2, 3], [6, 3, 4]])
# y = np.mean(x, axis=1)
# print(x)
# print(y)
# print(x - y[:, None])

s1 = [0, 2, 0, 3, 4]
s2 = [0, 0, 0, 5, 2]
zero_pos = [k for k in range(len(s1)) if s1[k] == 0 and s2[k] == 0]
print(zero_pos)
