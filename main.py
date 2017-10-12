from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from yolo_net import *
from scipy.ndimage.interpolation import zoom
from PIL import Image

image = Image.open("E:/Datasets/VOC2012/JPEGImages/2007_000027.jpg")
image.show()
image = image.resize((259,259))
image.show()
image = (np.asarray(image) / 255).reshape(1,259,259,3)
print(image)

is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape = [None, 259, 259, 3])

yolo_output = YOLO_network(x, is_training)
loss = loss(yolo_output)
#train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test = sess.run(yolo_output, feed_dict = {x : image, is_training : True})
    print(test.shape)
