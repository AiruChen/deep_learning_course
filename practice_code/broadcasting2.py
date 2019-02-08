import numpy as nu

A = nu.array([ [56.0, 0.0, 4.0, 68.0], [1.2, 104.0, 52.0, 8.0], [1.8, 135.0,
    99.0, 0.9]])

print(A)

cal = A.sum(axis=0) # axis=0 sum up amoung column, axis=1 sum up amoung raw
print(cal)

print(cal.reshape(1,4)) # .reshape make sure correct (raw, column) number

percentage = 100*A/cal.reshape(1,4) # A:3X4 cal:1X4 using broadcasting
print('percentage = ', percentage)
