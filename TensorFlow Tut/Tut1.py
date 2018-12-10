#!/usr/bin/env python

import tensorflow as tf
import numpy as np

b = tf.Variable(tf.zeros((100,)))
w = tf.Variable(tf.random_uniform((784, 100), -1, 1))
x = tf.placeholder(tf.float32, (100, 784))

h = tf.nn.relu(tf.matmul(x, w) + b)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(h, {x : np.random.random(100, 784)})