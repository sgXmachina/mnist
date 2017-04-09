import numpy as np
import tensorflow as tf
import tf_utils

from tensorflow.examples.tutorials.mnist import input_data

__author__ = 'shivam'

def main():
    """
    Runs a simple linear regression model on the mnist dataset.
    """

    # Load the mnist dataset. Class stores the train, validation and testing sets as numpy arrays. 
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Create a tensforlow session.
    sess = tf.InteractiveSession()

    # Create the computational graph. Start with creating placeholders for the input and output data.
    # Input placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape = [None, 784])
    # Output placeholder.
    labeled_data = tf.placeholder(tf.float32, shape = [None, 10])

    # Reshape input to a 4D tensor of [ -1 , width, height, channels]. -1 ensures the size remains consitent with 
    # the original size.
    image_shape = [-1,28,28,1]
    input_image = tf.reshape(input_placeholder, image_shape)

    # Create convolutional layers containing 2 convolutional layers and 1 fully connected layer.
    # Layer 1 computes 32 features for each 5x5 patch.
    conv1_weights = tf_utils.weight_variable([5,5,1,32])
    conv1_bias = tf_utils.bias_variable([32])
    # Apply ReLU activation and max pool.
    conv1_act = tf.nn.relu(tf_utils.conv2d(input_image,conv1_weights) + conv1_bias)
    conv1_pool = tf_utils.max_pool_2x2(conv1_act)

    # Layer 2 computes 64 features of 5x5 patch.
    conv2_weights = tf_utils.weight_variable([5,5,32,64])
    conv2_bias = tf_utils.bias_variable([64])
    # Apply ReLU activation and max pool.
    conv2_act = tf.nn.relu(tf_utils.conv2d(conv1_pool, conv2_weights) + conv2_bias)
    conv2_pool = tf_utils.max_pool_2x2(conv2_act)  

    # Add fully connected layers.
    fc1_weights = tf_utils.weight_variable([7*7*64,1024])
    fc1_bias = tf_utils.bias_variable([1024])  
    # Apply Relu activation to flattened conv2d pool layer.
    conv2_flat = tf.reshape(conv2_pool, [-1,7*7*64])
    fc1_act = tf.nn.relu(tf.matmul(conv2_flat,fc1_weights) + fc1_bias)

    # Add dropout before the readout layer.    
    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(fc1_act, keep_prob)

    # Add the readout layer for the 10 classes.
    readout_weights =  tf_utils.weight_variable([1024,10])
    readout_bias = tf_utils.bias_variable([10])
    readout_act = tf.matmul(dropout, readout_weights) + readout_bias

    # Cross entropy loss between the output labels and the model.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labeled_data, logits = readout_act))

    # Define the training step with a learning rate for gradient descent and our cross entropy loss.
    learning_rate = 1e-4
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Initialize all variables.
    sess.run(tf.global_variables_initializer())

    # Training model evaluation placeholders.
    # Define a placeholder for comparing equality between output and labels.
    predictions = tf.equal(tf.argmax(labeled_data,1), tf.argmax(readout_act,1))
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

    # Run the training for a n steps.
    steps = 10000
    batch_size = 50
    for step in xrange(steps):
        # Sample a batch from the mnist dataset.
        batch = mnist.train.next_batch(batch_size)
        # Create a dict of the data from the sampled batch and run one training step.
        train_step.run(feed_dict={input_placeholder:batch[0],labeled_data:batch[1], keep_prob:0.5})

        # Print the training error after every 100 steps.
        if step%100==0:
            train_accuracy = accuracy.eval(feed_dict={input_placeholder:batch[0],labeled_data:batch[1], keep_prob:1.0})
            print "Step: ",step," | Train Accuracy: ",train_accuracy


    print "Accuracy: ", accuracy.eval(feed_dict={input_placeholder:mnist.test.images, labeled_data:mnist.test.labels, keep_prob:1.0})    


if __name__ == "__main__":
    main()


