from __future__ import print_function


import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets
from dataset import Dataset

NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_CLASSES = 100
IMG_SIZE = 224
LEARNING_RATE = 0.01
DROP_KEEP_PROB = 0.5
TEST_BATCH_SIZE = 200 

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'full', 'the model used')
tf.flags.DEFINE_string('summaries_dir', './summaries', 'directory for summaries')
tf.flags.DEFINE_string('checkpoints_dir', './checkpoints', 'directory for checkpoints')


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


with tf.variable_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,
        logits=logits)
    loss = tf.reduce_mean(cross_entropy, name='loss')

    loss_count = tf.zeros(name='loss', shape=[TEST_BATCH_SIZE])
    b_loss = tf.reduce_mean(loss_count)

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
    sum1 = tf.reduce_sum(tf.cast(correct1, tf.float32))
    sum5 = tf.reduce_sum(tf.cast(correct5, tf.float32))
    accuracy1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
    accuracy5 = tf.reduce_mean(tf.cast(correct5, tf.float32))

    count1 = tf.zeros(name='count1', shape=[TEST_BATCH_SIZE])
    count5 = tf.zeros(name='count5', shape=[TEST_BATCH_SIZE])
    b_accuracy1 = tf.reduce_sum(count1) / 10000.0
    b_accuracy5 = tf.reduce_sum(count5) / 10000.0


init = tf.global_variables_initializer()
vgg_saver = tf.train.Saver(var_list=vgg_vars)
saver = tf.train.Saver(var_list=add_vars)

loss_summary = tf.summary.scalar('loss', loss)
b_loss_summary = tf.summary.scalar('train_summary', b_loss)
accuracy1_summary = tf.summary.scalar('accuracy1', b_accuracy1)
accuracy5_summary = tf.summary.scalar('accuracy5', b_accuracy5)

full_summary = tf.summary.merge([b_loss_summary, accuracy1_summary, accuracy5_summary])

num_iterations = train.num_samples // BATCH_SIZE

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    init.run()
    # vgg_saver.restore(sess, '/home/rriverag/cifar-100-python/vgg_16.ckpt')
    saver.restore(sess, FLAGS.checkpoints_dir + '/add_model.ckpt')

    feed_dict = {}

    for epoch in range(NUM_EPOCHS):
        for i in range(num_iterations):
            rgb_batch, y_batch = train.get_batch(BATCH_SIZE)
            feed_dict[rgb] = rgb_batch
            feed_dict[y] = y_batch
            sess.run(full_opt_op, feed_dict=feed_dict)
            saver.save(sess, FLAGS.checkpoints_dir + '/add_model.ckpt')
            # acc_train = accuracy.eval(feed_dict={rgb: rgb_batch, y: y_batch})

        # Train loss
        summary, train_loss = sess.run([loss_summary, loss],
                                       feed_dict=feed_dict)
        train_writer.add_summary(summary, epoch)

        # Test loss and accuracy
        temp = np.zeros([TEST_BATCH_SIZE, 3])
        for i in range(10000 // TEST_BATCH_SIZE):
            rgb_batch, y_batch = test.get_batch(TEST_BATCH_SIZE)
            feed_dict[rgb] = rgb_batch
            feed_dict[y] = y_batch
            temp[i, :] = sess.run([loss, sum1, sum5], feed_dict=feed_dict)

        summary, test_loss, test_accuracy1, test_accuracy5 = sess.run([full_summary, b_loss, b_accuracy1, b_accuracy5],
                                                                      feed_dict={loss_count: temp[:, 0],
                                                                                 count1: temp[:, 1],
                                                                                 count5: temp[:, 2]})

        """
        rgb_batch, y_batch = test.get_batch()
        feed_dict[rgb] = rgb_batch
        feed_dict[y] = y_batch
        summary, test_loss, test_accuracy1, test_accuracy5 = \
            sess.run([full_summary, loss, accuracy5, accuracy1],
                     feed_dict=feed_dict)
        """

        test_writer.add_summary(summary, epoch)

        test.clear_index()
        train.clear_index()
