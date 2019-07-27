import numpy as np


x = np.array([[1, 2, 3], [6, 3, 4]])
y = x.copy()
y[1, 2] = 999
print(x)
print(y)


