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
tf.flags.DEFINE_string('summaries_dir', None, 'directory for summaries')


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
    correct1 = tf.nn.in_top_k(logits, y, 1)
    correct5 = tf.nn.in_top_k(logits, y, 5)
    accuracy1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
    accuracy5 = tf.reduce_mean(tf.cast(correct5, tf.float32))

init = tf.global_variables_initializer()
vgg_saver = tf.train.Saver(var_list=vgg_vars)
saver = tf.train.Saver(var_list=add_vars)

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy1', accuracy1)
tf.summary.scalar('accuracy5', accuracy5)
merged_summary = tf.summary.merge_all()

num_iterations = train.num_samples // BATCH_SIZE

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    init.run()
    vgg_saver.restore(sess, '/home/rriverag/cifar-100-python/vgg_16.ckpt')

    feed_dict = {}

    for epoch in range(NUM_EPOCHS):
        for i in range(num_iterations):
            rgb_batch, y_batch = train.get_batch(BATCH_SIZE)
            feed_dict[rgb] = rgb_batch
            feed_dict[y] = y_batch
            sess.run(full_opt_op, feed_dict=feed_dict)
            saver.save(sess, 'saved/add_model.ckpt')
            # acc_train = accuracy.eval(feed_dict={rgb: rgb_batch, y: y_batch})

        # Train loss
        summary, train_loss = sess.run([merged_summary, loss],
                                       feed_dict=feed_dict)
        train_writer.add_summary(summary, epoch)

        # Test loss and accuracy
        rgb_batch, y_batch = test.get_batch()
        feed_dict[rgb] = rgb_batch
        feed_dict[y] = y_batch
        summary, test_loss, test_accuracy1, test_accuracy5 = \
            sess.run([merged_summary, loss, accuracy5, accuracy1],
                     feed_dict=feed_dict)
        test_writer.add_summary(summary, epoch)

        test.clear_index()
        train.clear_index()
