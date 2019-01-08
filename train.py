from __future__ import print_function


import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets
from dataset import Dataset


NUM_EPOCHS = 100
BATCH_SIZE = 256
NUM_CLASSES = 100
IMG_SIZE = 224
LEARNING_RATE = 0.00001

train = Dataset('cifar-100-python/train')

rgb = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
y = tf.placeholder(tf.int64, shape=[None])

with tf.name_scope('model'):
    logits, __ = tf.contrib.slim.nets.vgg.vgg_16(rgb, num_classes=NUM_CLASSES)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                   logits=logits)
    loss = tf.reduce_mean(cross_entropy, name='loss')

with tf.name_scope('optimizer'):
    first_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scope/') 
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    full_opt_op = optimizer.minimize(loss)
    first_opt_op = optimizer.minimize(loss, var_list=first_layers)

saver = tf.train.Saver()

num_iterations = train.num_samples // BATCH_SIZE

with tf.Session() as sess:
    saver.restore(sess, 'cifar-100-python/vgg_16.ckpt')
    for i in range(num_iterations):
        rgb_batch, y_batch = train.get_batch(BATCH_SIZE)
        rgb_batch = tf.image.resize_images(rgb_batch, [IMG_SIZE, IMG_SIZE])
        sess.run(full_opt_op, feed_dict={rgb: rgb_batch, y: y_batch})
        saver.saver(sess, 'saved/model.ckpt')
