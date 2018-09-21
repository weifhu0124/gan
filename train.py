import tensorflow as tf
from generator import generator
from discriminator import discriminator
from loader import load_data
import numpy as np


# hyperparameters
learning_rate_g = 0.0001
learning_rate_d = 0.0003
noise_dim = 100
batch_size = 50
discriminator_pre_train_loop = 300
train_loop = 2000

# load data
mnist = load_data()
print('data loaded')

# placeholder for noise
noise_placeholder = tf.placeholder(tf.float32, [None, noise_dim], name='noise_placeholder')
# placeholder for real image
x_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x_placeholder')

# generate new images
generated = generator(noise_placeholder, noise_dim)

# discriminator for real MNIST dataset
d_real = discriminator(x_placeholder)
# discriminator for generated data
d_fake = discriminator(generated, reuse=True)

# define loss function
# real loss -> probability as close to 1 as possible
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
# generated loss -> probability as close to 0 as possible
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
# generator loss -> wants to let discriminator output 1 as much as possible for generated images
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

# get trainable variables
variables = tf.trainable_variables()
discriminator_var = [var for var in variables if 'd_' in var.name]
generator_var = [var for var in variables if 'g_' in var.name]

# optimizer
discriminator_train_real = tf.train.AdamOptimizer(learning_rate=learning_rate_d)\
    .minimize(d_loss_real, var_list=discriminator_var)
discriminator_train_fake = tf.train.AdamOptimizer(learning_rate=learning_rate_d)\
    .minimize(d_loss_fake, var_list=discriminator_var)
generator_train = tf.train.AdamOptimizer(learning_rate=learning_rate_g).minimize(g_loss, var_list=generator_var)

tf.get_variable_scope().reuse_variables()
# open a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# restore model from previous training - we can train 100 loops at first, then
# we run another 100 loops. With following restore code we train a total of 200 loops
# COMMENT OUT IF YOU WANT TO TRAIN FROM BEGINNING
saver.restore(sess, 'model/mnist_gan.ckpt')

print('pre-train discriminator...')
# pre-train discriminator
for i in range(0, discriminator_pre_train_loop):
    # create noise vector
    noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
    real_images = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    # pretrain on both real and fake images
    _, __, d_loss_real, d_loss_fake = sess.run([discriminator_train_real, discriminator_train_fake,
                                                d_loss_real, d_loss_fake],
                                               feed_dict={x_placeholder: real_images, noise_placeholder: noise})
    # need to cast back to tensor, otherwise there would be an error
    d_loss_real = tf.cast(d_loss_real, tf.float32)
    d_loss_fake = tf.cast(d_loss_fake, tf.float32)

print('training...')
# training
for i in range(0, train_loop):
    noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
    real_images = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __, d_loss_real, d_loss_fake = sess.run([discriminator_train_real, discriminator_train_fake,
                                                d_loss_real, d_loss_fake],
                                               feed_dict={x_placeholder: real_images, noise_placeholder: noise})
    # train generator
    noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
    sess.run(generator_train, feed_dict={x_placeholder: real_images, noise_placeholder: noise})
    d_loss_real = tf.cast(d_loss_real, tf.float32)
    d_loss_fake = tf.cast(d_loss_fake, tf.float32)
    # save each 100 loops
    if i % 100 == 0 and i != 0:
        print(str(i))
        saver.save(sess, 'model/mnist_gan.ckpt')

print('training completed')
saver.save(sess, 'model/mnist_gan.ckpt')
