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
gt = (174/486, 101/500, 349/486, 351/500)
#loss = loss_op(yolo_output, gt)
#train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
test_op = test_op(yolo_output, gt)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iterations = 1
    for _ in range(iterations):
        #sess.run(train_op, feed_dict = {x : image, is_training : True})
        cost = sess.run(test_op, feed_dict = {x : image, is_training : True})
        print(cost)
