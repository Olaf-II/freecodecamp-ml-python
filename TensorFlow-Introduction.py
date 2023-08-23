import tensorflow as tf
import numpy as np

string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.562, tf.float64)

rank1_tensor = tf.Variable(["Test", "Ok", "Tim"], tf.string)   # Rank 1 because it's one array.
rank2_tensor = tf.Variable([["Test", "Ok", "Nick"], ["Test", "Ok", "Nick"]], tf.string)   # Rank 2 because it's two arrays. Must ensure all lists are kept inside a "master array"

print(tf.rank(rank2_tensor))   # Prints the rank, shape and dtype (data type)

print(rank2_tensor.shape)   # Prints shape
print(rank1_tensor.shape)   # Prints shape
print(string.shape)   # Prints shape

tensor1 = tf.ones([1, 2, 3])   # Makes a tensor in dimensions 1, 2, 3 which is filled with ones
print(tensor1)   # Prints tensor
tensor2 = tf.reshape(tensor1, [2, 3, 1])   # Reshapes the tensor to a new shape. This signifies 2 lists, inside of each of those lists is 3 lists, and inside of each of those lists is 1 element.
print(tensor2)   # Prints tensor
tensor3 = tf.reshape(tensor2, [3, -1])   # -1 calculates the size required based on existing dimensions. Since it's 6 elements, it must be 2. All shape elements multiply to the number of elements.
print(tensor3)

t = tf.zeros([5, 5, 5, 5])
print(t)
t = tf.reshape(t, [625])
print(t)
t = tf.reshape(t, [125, -1])
print(t)