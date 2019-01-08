from __future__ import print_function


import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets
from dataset import Dataset


NUM_EPOCHS = 100
BATCH_SIZE = 128
NUM_CLASSES = 100
IMG_SIZE = 224
LEARNING_RATE = 0.001
DROP_KEEP_PROB = 0.5

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'full', 'the model used')


train = Dataset('/home/rriverag/cifar-100-python/train')
test = Dataset('/home/rriverag/cifar-100-python/test')

rgb = tf.placeholder(tf.float32,
                     shape=[None, 32, 32, 3],
                     name='rgb')
y = tf.placeholder(tf.int64,
                   shape=[None],
                   name='y')

resized_rgb = tf.image.resize_images(rgb, [IMG_SIZE, IMG_SIZE])

__, end_points = tf.contrib.slim.nets.vgg.vgg_16(resized_rgb)

# print(end_points)

net = end_points['vgg_16/fc7']

with tf.variable_scope('add'):
    net = tf.layers.dropout(net, rate=DROP_KEEP_PROB, name='dropout1')
    net = tf.layers.conv2d(net,
                           NUM_CLASSES,
                           [1, 1],
                           activation=None,
                           name='conv1')
    logits = tf.squeeze(net, name='squeezed1')

net2 = end_points['vgg_16/fc5']

with tf.variable_scope('model2'):
    pass

with tf.variable_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,
        logits=logits)
    loss = tf.reduce_mean(cross_entropy, name='loss')

# print([n.name for n in tf.get_default_graph().as_graph_def().node])

with tf.name_scope('train'):
    vgg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg_16')
    add_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'add')
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    full_opt_op = optimizer.minimize(loss)
    add_opt_op = optimizer.minimize(loss, var_list=add_vars)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
vgg_saver = tf.train.Saver(var_list=vgg_vars)
saver = tf.train.Saver(var_list=add_vars)

tf.summary.scalar('train_loss', train_loss)
tf.summary.scalar('test_loss', test_loss)
tf.summary.scalar('test_accuracy_top1', test_accuracy1)
tf.summary.scalar('test_accuracy_top5', test_accuracy5)

num_iterations = train.num_samples // BATCH_SIZE

with tf.Session() as sess:
    init.run()
    vgg_saver.restore(sess, '/home/rriverag/cifar-100-python/vgg_16.ckpt')
    for epoch in range(NUM_EPOCHS):
        for i in range(num_iterations):
            rgb_batch, y_batch = train.get_batch(BATCH_SIZE)
            sess.run(full_opt_op, feed_dict={rgb: rgb_batch, y: y_batch})
            saver.save(sess, 'saved/add_model.ckpt')
            # acc_train = accuracy.eval(feed_dict={rgb: rgb_batch, y: y_batch})
            rgb_batch, y_batch = test.get_batch(256)
            acc_test = accuracy.eval(feed_dict={rgb: rgb_batch, y: y_batch})
            msg = 'epoch{} ({}/{}) Train accuracy: {} - Test accuracy: {}\n'.format(epoch,
                                                                                    i,
                                                                                    num_iterations,
                                                                                    acc_train,
                                                                                    acc_test)
            # f.write(msg)
            print(msg)
            test.clear_index()
        train.clear_index()
