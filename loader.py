from tensorflow.examples.tutorials.mnist import input_data

def load_data():
    return input_data.read_data_sets('MNIST_data/')
