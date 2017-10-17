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
image = (np.asarray(image) / 255).reshape(259,259,3)
image = np.array([image,image])
print(image)
print(image.shape)

is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape = [None, 259, 259, 3])
gt = tf.placeholder(tf.float32, shape = [None, None, 24]) # shape(batch_size, max_num_objects, 24)    ### xmin, ymin, xmax, ymax, class prediction
gt_num_objects = tf.placeholder(tf.int32, shape = [None]) #shape(batch_size)
batch_size = tf.placeholder(tf.int32)

yolo_tensor, class_tensor = YOLO_network(x, is_training)
gt_im1 = np.array([[[174/486, 101/500, 349/486, 351/500, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [      0,       0,       0,       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                    [[174/486, 101/500, 349/486, 351/500, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [174/486, 101/500, 349/486, 351/500, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
num_objects = [1, 2]
print(gt_im1)
print(gt_im1.shape)
#loss = loss_op(yolo_output, gt)
#train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
loss_op = loss_op(yolo_tensor, class_tensor, gt, gt_num_objects, batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iterations = 1
    for _ in range(iterations):
        #sess.run(train_op, feed_dict = {x : image, is_training : True})
        cost = sess.run(loss_op, feed_dict = {x : image, gt : gt_im1, gt_num_objects : num_objects, batch_size : 2, is_training : True})
        print(cost)
