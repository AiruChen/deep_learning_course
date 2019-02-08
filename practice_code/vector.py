import numpy as np

a = np.random.randn(5)  # rank 1 array

print(a)
print(a.shape)
print(a.T)  # here, you find a = a.T
print(np.dot(a,a.T))
# Don't call rank 1 array


# =======================================
# use the column or raw matrix
a = np.random.randn(5,1)    # define as raw vector
#a = np.random.randn(1,5)    # define as column vector

print(a)
print(a.shape)
print(a.T)
print(np.dot(a,a.T))
a = a.reshape(1,5)
print(a)
assert(a.shape == (5,1)) # make sure this is the vertor shape you want
