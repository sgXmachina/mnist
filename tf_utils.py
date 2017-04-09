"""
Utility functions for working on the mnist datset.
"""
import tensorflow as tf

__author__ = 'shivam'


def weight_variable(shape):
    """
    Initializes a weight tensor of the given shape with non-zero weights.
    :param shape  numpy array     Contains the shape of the tf.Variable to be initialized, i.e.,
                                a vector of 1X10 has shape = [None, 10]. 

    :return tf.Variable           tf tensor of given shape.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Initializes a bias tensor of given shape.
    :param shape  numpy array     Contains the shape of the tf.Variable to be initialized, i.e.,
                                  a vector of 1X10 has shape = [None, 10]. 
    
    :return tf.Variable           tf tensor of given shape.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(input_tensor,
           kernel_tensor,
           strides=[1,1,1,1],
           padding='SAME'):
    """
    Performs 2D convolutions on the input tensor.
    :param input_tensor     Tensor.                  Must be one of the following types: half, float32, float64.
    :param kernel_tensor    Tensor.                  Must have the same type as input.
    :param strides          list of ints.            1-D of length 4. The stride of the sliding window for
                                                     each dimension of input. Must be in the same order as the 
                                                     dimension specified with format.
    :param padding          string ("SAME"|"VALID")  The type of padding algorithm to use.
    
    :return Tensor of the same type as input.
    """

    return tf.nn.conv2d(input_tensor, kernel_tensor, strides=strides, padding=padding)

def max_pool_2x2(input_tensor,
                 ksize = [1,2,2,1],
                 strides=[1,2,2,1],
                 padding='SAME'):
    """
    Performs max pooling operation on the input vector.

    :param value        4-D Tensor               Should have shape [batch, height, width, channels] and type tf.float32.
    :param ksize        list of ints             The size of the window for each dimension of the input tensor. Should have length >= 4.
    :param strides      list of ints             The stride of the sliding window for each dimension of the input tensor. length >= 4.
    :param padding      string ("SAME"|"VALID")  The type of padding algorithm to use.

    :return Tensor      The max pooled output tensor.
    """
    return tf.nn.max_pool(input_tensor, ksize=ksize,
                        strides=strides, padding=padding)