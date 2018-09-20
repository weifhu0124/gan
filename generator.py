import tensorflow as tf

filter_num = 56*56

def generator(noise, noise_dim):
    weight1 = tf.get_variable('g_weight1', [noise_dim, filter_num], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.01))
    bias1 = tf.get_variable('g_bias1', [filter_num], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
    tensor1 = tf.matmul(noise, weight1) + bias1
    tensor1 = tf.reshape(tensor1, [-1, 56, 56, 1])
    tensor1 = tf.contrib.layers.batch_norm(tensor1, epsilon=1e-5, scope='g_bias1')
    tensor1 = tf.nn.relu(tensor1)

    # convolution 3x3 kernel size
    # use noise_dim / 2 filters
    weight2 = tf.get_variable('g_weight2', [3, 3, 1, noise_dim/2],
                              initializer=tf.truncated_normal_initializer(stddev=0.01))
    bias2 = tf.get_variable('g_bias2', [noise_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.01))
    tensor2 = tf.nn.conv2d(input=tensor1, filter=weight2, strides=[1, 2, 2, 1], padding='SAME')
    tensor2 = tensor2 + bias2
    tensor2 = tf.contrib.layers.batch_norm(tensor2, epsilon=1e-5, scope='g_bias2')
    tensor2 = tf.nn.relu(tensor2)
    # use bicubic to enlarge
    tensor2 = tf.image.resize_images(tensor2, [56, 56], method=tf.image.ResizeMethod.BICUBIC)

    # convolution 3x3 kernel size
    # use noise_dim / 4 filters
    weight3 = tf.get_variable('g_weight3', [3, 3, noise_dim/2, noise_dim/4],
                              initializer=tf.truncated_normal_initializer(stddev=0.01))
    bias3 = tf.get_variable('g_bias3', [noise_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.01))
    tensor3 = tf.nn.conv2d(input=tensor2, filter=weight3, strides=[1, 2, 2, 1], padding='SAME')
    tensor3 = tensor3 + bias3
    tensor3 = tf.contrib.layers.batch_norm(tensor3, epsilon=1e-5, scope='g_bias3')
    tensor3 = tf.nn.relu(tensor3)
    # use bicubic to enlarge
    tensor3 = tf.image.resize_images(tensor3, [56, 56], method=tf.image.ResizeMethod.BICUBIC)

    # final convolution to generate image
    weight4 = tf.get_variable('g_weight4', [1, 1, noise_dim/4, 1],
                              initializer=tf.truncated_normal_initializer(stddev=0.01))
    bias4 = tf.get_variable('g_bias4', [1], initializer=tf.truncated_normal_initializer(stddev=0.01))
    tensor4 = tf.nn.conv2d(input=tensor3, filter=weight4, strides=[1, 2, 2, 1], padding='SAME')
    tensor4 = tensor4 + bias4
    tensor4 = tf.sigmoid(tensor4)
    return tensor4
