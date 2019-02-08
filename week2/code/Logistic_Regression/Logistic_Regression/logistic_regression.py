import numpy as np
import matplotlib.pyplot as plt
import h5py
# import scipy
# from PIL import Image
# from scipy import ndimage
from read import load_dataset
import functions as fn

# loading the data ( cat/non-cat)
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

# # check data shape
# print('train_set_x_orig.shape = ', train_set_x_orig.shape)
# print('train_set_y_orig.shape = ', train_set_y_orig.shape)
# print('test_set_x_orig.shape = ', test_set_x_orig.shape)
# print('test_set_y_orig.shape = ', test_set_y_orig.shape)
#
# # example of picture
# for index in range(1,10):
#     plt.subplot( 2,5,index)
#     plt.imshow(train_set_x_orig[index])
#     print("y = " + str(train_set_y_orig[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y_orig[:,index])].decode("utf-8") +"' picpicture.")
# plt.show()

m_train = train_set_x_orig.shape[0] # number of train example
m_test = test_set_x_orig.shape[0]   # number of test example
num_px = train_set_x_orig.shape[1]  # = height = width pixel of a training image
# print('m_train', m_train)
# print('m_test', m_test)
# print('num_px', num_px)

# flatten the training example into a column vector of shape(num_px*num_px*3,1) = (height*width*rgb,1)
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
# print('train_set_x_flatten.shape', train_set_x_flatten.shape)
# print('test_set_x_flatten.shape', test_set_x_flatten.shape)

# standardize dataset, which means to divide every row of the dataset by 255(maximun pixel)
# print(train_set_x_flatten)
train_set_x = train_set_x_flatten/255.   # use broadcast
test_set_x = test_set_x_flatten/255.     # use broadcast
# print(train_set_x)

# Key step of Logistic regression
# 1 Initialize the parameters of the model
# 2 Learn the parameters for the model by minimizing the cost
# 3 Use the learned parameters to make predections
# 4 Analyse the results and conclude
# go chech functions.py
d = fn.model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# check example
# example of picture
for index in range(1,10):
    plt.subplot( 2,5,index)
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    print("y = " + str(test_set_y_orig[0,index]) + ", you predict that is a '" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +"' picpicture.")
# plt.show()

# plot learning curve
# print(np.squeeze(d["costs"]))
cost = np.squeeze(d["costs"])
plt.figure()
plt.plot(cost)
plt.ylabel('cost')
plt.xlabel('iterations (per hungreds)')
plt.title("Learning rate ="+ str(d["learning_rate"]))
plt.show()
