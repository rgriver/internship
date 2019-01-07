from __future__ import print_function



import tensorflow as tf
import numpy as np
import pickle
import vgg
from dataset import Dataset



NUM_EPOCHS = 100
BATCH_SIZE = 512
NUM_CLASSES = 100
NUM_ITERATIONS = NUM_TRAIN_SAMPLES // BATCH_SIZE
LEARNING_RATE = 0.00001

    
train = Dataset('cifar-100-python/train')

rgb = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.int64, shape=[None])

with tf.name_scope('model'):
    logits, __ = vgg_16(rgb, num_classes=NUM_CLASSES)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name='loss')

with tf.name_scope('optimizer'):
    first_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scope/') 
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    full_opt_op = optimizer.minimize(loss)
    first_opt_op = optimizer.minimize(loss, var_list=first_layers)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'cifar-100-python/vgg_16.ckpt')
    for i in range(NUM_ITERATIONS):
        rgb_batch, y_batch = train.get_batch(BATCH_SIZE)
        sess.run(opt_op, feed_dict={rgb: rgb_batch, y: y_batch})
        saver.saver(sess, 'saved/model.ckpt')



