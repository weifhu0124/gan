import tensorflow as tf
import numpy as np
from generator import generator
from matplotlib import pyplot as plt

noise_dim = 100
batch_size = 50
result_path = 'result/'

# restore the session from training
tf.reset_default_graph()
# construct generator
noise_placeholder = tf.placeholder(tf.float32, [None, noise_dim])
generated_images = generator(noise_placeholder, noise_dim)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'model/mnist_gan.ckpt')
    noise = np.random.normal(0, 1, size=[batch_size, noise_dim])

    generate_mnist = sess.run(generated_images, feed_dict={noise_placeholder:noise})
    # save result to folder
    for j in range(batch_size):
        filename = result_path + str(j) + ".png"
        plt.imsave(filename, generate_mnist[j].reshape([28, 28]))
