import numpy as np

# to keep the example easy to view, we use 10 examples of pictures.  Each
# picture is of size 4x4 pixels, and each pixel contains 3 (r,g,b) values
a=np.arange(10*4*4*3).reshape(10,4,4,3)
print(a.shape)
print(a)

# ===========================================================
# a -1 argument tells numpy to figure out the dimensions of reshaped axis
aflat=a.reshape(a.shape[0], a.shape[1], a.shape[2], -1)
print(aflat.shape)
print(aflat)

# ===========================================================
# flatten the innermost two axis (r,g,b values in each pixel row). 4x3 gets
# flattened to 12 color values
aflat=a.reshape(a.shape[0],a.shape[1],-1)
print(aflat.shape)
print(aflat)

# ===========================================================
# flatten the innermost three axis (r,g,b values in each pixel row, reading
# left to right and top to bottom). 4x4x3 gets flattened to 48 values.  this
# operation flattens each individual image
aflat=a.reshape(a.shape[0],-1)
print(aflat.shape)
print(aflat)
# at this point, the rows have 'examples' (the training or test cases) and
# columns have the 'features' (the color values).  to get the features in rows
# and examples in columns, we transpose the matrix using the .T method
aflatt=aflat.T
print(aflatt.shape)
print(aflatt)

