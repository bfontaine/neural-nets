# -*- coding: UTF-8 -*-

from random import randrange

import tensorflow as tf
import numpy as np

s = tf.InteractiveSession()

BITS = 64

def get_bits(n):
    bits = np.zeros([BITS])

    # pretty ugly
    for i, c in enumerate(reversed(np.binary_repr(n))):
        bits[ BITS-i-1 ] = float(c)

    return bits

x = tf.placeholder(tf.float32, shape=[None, BITS])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.zeros([BITS,2]))
b = tf.Variable(tf.zeros([2]))

s.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 1e-50),
                               reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    even = [get_bits(randrange(0, 1e12, 2)) for _ in range(100)]
    odd =  [get_bits(randrange(1, 1e12, 2)) for _ in range(100)]

    train_step.run(feed_dict={x: even, y_: [ [0, 1] ]})
    train_step.run(feed_dict={x: odd, y_: [ [1, 0] ]})

def predict(n):
    p = s.run(y, {x: [ get_bits(n) ]})

    for i, label in enumerate(("odd", "even")):
        prob = p[0, i]
        if prob > 0.9:
            return {"label": label, "confidence": prob}
