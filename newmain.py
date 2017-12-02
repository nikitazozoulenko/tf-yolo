from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import threading
import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from yolo_net import *
from VOC_queue import *

MAX_NUM_OBJECTS_VOC = 56
batch_size = 2
directory = "E:/Datasets/VOC2012"

classtoint_dict = {"aeroplane" : 0,
                   "bicycle" : 1,
                   "bird" : 2,
                   "boat" : 3,
                   "bottle" : 4,
                   "bus" : 5,
                   "car" : 6,
                   "cat" : 7,
                   "chair" : 8,
                   "cow" : 9,
                   "diningtable" : 10,
                   "dog" : 11,
                   "horse" : 12,
                   "motorbike" : 13,
                   "person" : 14,
                   "pottedplant" : 15,
                   "sheep" : 16,
                   "sofa" : 17,
                   "train" : 18,
                   "tvmonitor" : 19}

is_training = tf.placeholder(tf.bool)


image_array_op, gt_array_op, num_objects_op = get_next_batch()

box_tensor, confidence_tensor, class_tensor = inference(image_array_op, tf.constant(True))

def testop():
    box_index = 0
    cell_x = 0
    cell_y = 3
    # one_hot0 = tf.one_hot(indices = box_index,
    #                       depth = 2,
    #                       on_value = cell_x,
    #                       off_value = -1,
    #                       axis = 0)
    # one_hot1 = tf.one_hot(indices = one_hot0,
    #                       depth = 7,
    #                       on_value = cell_y,
    #                       off_value = -1,
    #                       axis = -1)
    # one_hot2 = tf.one_hot(indices = one_hot1,
    #                                  depth = 7,
    #                                  on_value = 1.0,
    #                                  off_value = 0.0,
    #                                  axis = -1)
    one_hot1 = tf.one_hot(indices = cell_x,
                          depth = 7,
                          on_value = cell_y,
                          off_value = -1,
                          axis = -1)
    one_hot2 = tf.one_hot(indices = one_hot1, depth = 7)
    return one_hot1, one_hot2
testop = testop()
loss_op = loss(box_tensor, confidence_tensor, class_tensor, gt_array_op, num_objects_op)
train_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_op)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    # loss = sess.run(loss_op)
    # print(loss)
    # for _ in range(10):
    #     loss, _ = sess.run([loss_op, train_op])
    #     print(loss)
    # result = sess.run(testop)
    # print(result)

    for _ in range(3):
        image, gt, num_objects = sess.run([image_array_op, gt_array_op, num_objects_op])
        show_image = (image[0]*255).astype(np.uint8)
        im = Image.fromarray(show_image)
        im.show()
        dr = ImageDraw.Draw(im)
        for i in range(num_objects[0,0]):
            coords = (gt[0, i, 0:4] * 259).astype(int)
            border_size = 5
            for j in range(border_size):
                coords_iter = (coords[0]+j, coords[1]+j, coords[2]-j, coords[3]-j)
                dr.rectangle(coords_iter, outline = "red")
        im.show()



    coord.request_stop()
    coord.join(threads)


# show_image = (image[0]*255).astype(np.uint8)
# im = Image.fromarray(show_image)
# im.show()
# dr = ImageDraw.Draw(im)
# for i in range(num_objects[0,0]):
#     coords = (gt[0, i, 0:4] * 259).astype(int)
#     coords = (coords[0], coords[1], coords[2], coords[3])
#     dr.rectangle(coords, outline = "red")
# im.show()
