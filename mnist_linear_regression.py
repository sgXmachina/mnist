import numpy as np
import tensorflow as tf
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

    # Capture the model parameters.
    # Create a variable for weights.
    layer1_wts = tf.Variable(tf.zeros([784,10]))
    # Create a variable for biases.
    bias = tf.Variable(tf.zeros([10]))

    # Initialize all variables.
    sess.run(tf.initialize_all_variables())
    # sess.run(tf.global_variables_initializer())

    # Simple regression model where the output is a weighted combination of the input +bias.
    output = tf.matmul(input_placeholder, layer1_wts) + bias

    # Cross entropy loss between the output labels and the model.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labeled_data, logits = output))

    # Define the training step with a learning rate for gradient descent and our cross entropy loss.
    learning_rate = 0.4
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # Run the training for a n steps.
    steps = 1000
    batch_size = 100
    for _ in xrange(steps):
        # Sample a batch from the mnist dataset.
        batch = mnist.train.next_batch(batch_size)
        # Create a dict of the data from the sampled batch and run one training step.
        train_step.run(feed_dict={input_placeholder:batch[0],labeled_data:batch[1]})


    # Evaluate the trained model.
    # Define a placeholder for comparing equality between output and labels.
    predictions = tf.equal(tf.argmax(labeled_data,1), tf.argmax(output,1))
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

    print "Accuracy: ", accuracy.eval(feed_dict={input_placeholder:mnist.test.images, labeled_data:mnist.test.labels})    


if __name__ == "__main__":
    main()


