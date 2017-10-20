from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from yolo_net import *
from scipy.ndimage.interpolation import zoom
from PIL import Image

image = Image.open("E:/Datasets/VOC2012/JPEGImages/2007_000027.jpg")
image = image.resize((259,259))
image = (np.asarray(image) / 255).reshape(259,259,3)

image2 = Image.open("E:/Datasets/VOC2012/JPEGImages/2007_000032.jpg")
image2 = image2.resize((259,259))
image2 = (np.asarray(image2) / 255).reshape(259,259,3)

images = np.array([image,image2])
print(images.shape)

is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape = [None, 259, 259, 3])
gt = tf.placeholder(tf.float32, shape = [None, None, 5]) # shape(batch_size, max_num_objects, 5)    ### xmin, ymin, xmax, ymax, class prediction
gt_num_objects = tf.placeholder(tf.int32) #shape(batch_size)
batch_size = tf.placeholder(tf.int32)

box_tensor, confidence_tensor, class_tensor = YOLO_network(x, is_training)
gt_im1 = np.array([[[174/486, 101/500, 349/486, 351/500, 0],
                    [174/486, 101/500, 349/486, 351/500, 0],
                    [174/486, 101/500, 349/486, 351/500, 0],
                    [174/486, 101/500, 349/486, 351/500, 0]],

                    [[104/500, 78/281, 375/500, 183/281, 17],
                     [133/500, 88/281, 197/500, 123/281, 17],
                     [195/500, 180/281, 213/500, 229/281, 0],
                     [ 26/500, 189/281,  44/500, 238/281, 0]]])
num_objects = [1,4]

# gt_im1 = np.array([[[104/500, 78/281, 375/500, 183/281, 17],
#                       [133/500, 88/281, 197/500, 123/281, 17],
#                       [195/500, 180/281, 213/500, 229/281, 0],
#                       [ 26/500, 189/281,  44/500, 238/281, 0]]])


expected = np.asarray([gt_im1[:,:,0],
gt_im1[:,:,1],
gt_im1[:,:,2] - gt_im1[:,:,0],
gt_im1[:,:,3] - gt_im1[:,:,1]]).T
print("expected:")
print(expected)

# gt_im1 = np.array([[[174/486, 101/500, 349/486, 351/500, 0]]])
#num_objects = [4,-1]
#print(gt_im1)
#print(gt_im1.shape)

loss_op = loss(box_tensor, confidence_tensor, class_tensor, gt, gt_num_objects, batch_size)
train_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_op)

detection_op = detect_objects(box_tensor, confidence_tensor, class_tensor)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iterations = 10
    print("cost: ", sess.run(loss_op, feed_dict = {x : images, gt : gt_im1, gt_num_objects : num_objects, batch_size : 2, is_training : True}))
    boxes, classes, probs = sess.run(detection_op, feed_dict = {x : images[1:2, :, :, :], is_training : False})
    print("probs before", probs)
    for _ in range(iterations):
        #sess.run(train_op, feed_dict = {x : image, is_training : True})
        sess.run(train_op, feed_dict = {x : images, gt : gt_im1, gt_num_objects : num_objects, batch_size : 2, is_training : True})
    print(sess.run(loss_op, feed_dict = {x : images, gt : gt_im1, gt_num_objects : num_objects, batch_size : 2, is_training : True}))
    # boxes, classes, probs = sess.run(detection_op, feed_dict = {x : images[0:1, :, :, :], is_training : True})
    # print("IMAGE1")
    # print(classes)
    # boxes, classes, probs = process_boxes(0.7, boxes, classes, probs)
    # for box in boxes: print("box", box)
    # print(classes)
    # print(probs)
    # print("IMAGE1END")

    print("IMAGE2")
    print("DO RECTANGLE BNDBOX IN OPENCV????? RESEARCH THAT")
    boxes, classes, probs = sess.run(detection_op, feed_dict = {x : images[1:2, :, :, :], is_training : False})
    print("classes",classes)
    print("probs",probs)
    boxes, classes, probs = process_boxes(0.4, boxes, classes, probs)
    for box in boxes: print("box", box)
    print(classes)
    print(probs)
    print("IMAGE2END")
