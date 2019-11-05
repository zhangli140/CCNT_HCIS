# x = 0.06
# step = 0
# while True:
#     print x, step
#     step += 1
#     x = x * 0.9
#     raw_input()

import numpy as np
import tensorflow as tf
np.random.seed(1)
a = np.arange(0,100)
print(np.random.rand(4))
print(np.random.rand(4))

a = tf.Variable((3,4),dtype=tf.int8)
print tf.shape(a)
# with tf.Session() as sess:
#     sess.run(a)

print (a.shape)
print(np.random.rand(4))
