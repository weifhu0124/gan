import tensorflow as tf

def discriminator(image, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
        # input 28x28x1
        # convolution layer 5x5x32
        weight1 = tf.get_variable('d_weight1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias1 = tf.get_variable('d_bias1', [32], initializer=tf.constant_initializer(0))
        tensor1 = tf.nn.conv2d(input=image, filter=weight1, strides=[1, 1, 1, 1], padding='SAME')
        tensor1 = tensor1 + bias1
        # activation
        tensor1 = tf.nn.relu(tensor1)
        # average pooling
        tensor1 = tf.nn.avg_pool(tensor1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # output 14x14x32

        # convolution layer 5x5x64
        weight2 = tf.get_variable('d_weight2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias2 = tf.get_variable('d_bias2', [64], initializer=tf.constant_initializer(0))
        tensor2 = tf.nn.conv2d(input=tensor1, filter=weight2, strides=[1, 1, 1, 1], padding='SAME')
        tensor2 = tensor2 + bias2
        # activation
        tensor2 = tf.nn.relu(tensor2)
        # average pooling
        tensor2 = tf.nn.avg_pool(tensor2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # output 7x7x64

        # fully connected layer
        weight3 = tf.get_variable('d_weight3', [7*7*64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias3 = tf.get_variable('d_bias3', [1024], initializer=tf.constant_initializer(0))
        tensor2_flat = tf.reshape(tensor2, [-1, 7*7*64])
        tensor3 = tf.matmul(tensor2_flat, weight3) + bias3
        tensor3 = tf.nn.relu(tensor3)

        # fully connected layer
        weight4 = tf.get_variable('d_weight4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
        bias4 = tf.get_variable('d_bias4', [1], initializer=tf.constant_initializer(0))
        tensor4 = tf.matmul(tensor3, weight4) + bias4
        return tensor4
