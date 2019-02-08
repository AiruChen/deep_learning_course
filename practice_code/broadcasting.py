import numpy as np

X = np.array([[1, 2, 3],[4, 5, 6], [ 7, 8, 9]])
w = np.array([[1], [2], [3]])
b = 1

Z = np.dot(w.T, X) + b

print('X = ',X)
print('w = ',w)
print('b = ',b)
print('Z = ',Z)
