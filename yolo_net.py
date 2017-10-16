from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from resnet import *

def YOLO_network(x, is_training, confidence_threshhold = 0.5):
    #Nonex259x259x3 input image
    #model used is ResNet-18, modified to fit the tiny imagenet dataset
    with tf.variable_scope("conv1"):
        with tf.variable_scope("h1_conv_bn"):
            x = conv_wrapper(x, shape = [7,7,3,64], strides = [1, 2, 2, 1], padding = "VALID")
            x = tf.nn.max_pool(x, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "VALID")
            x = bn_wrapper(x, is_training)
            x = tf.nn.relu(x)

    with tf.variable_scope("conv2_x"):
        # 2 residual blocks, 64
        channels = 64
        with tf.variable_scope("residual_block_1"):
            x = residual_block(x, channels, is_training)
        with tf.variable_scope("residual_block_2"):
            x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv3_x"):
        # 2 residual blocks, 128
        channels = 128
        with tf.variable_scope("residual_block_1"):
            x = residual_block_reduce_size(x, channels, is_training)
        with tf.variable_scope("residual_block_2"):
            x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv4_x"):
        # 2 residual blocks, 192
        channels = 192
        with tf.variable_scope("residual_block_1"):
            x = residual_block_reduce_size(x, channels, is_training)
        with tf.variable_scope("residual_block_2"):
            x = residual_block(x, channels, is_training)

    with tf.variable_scope("conv5_x"):
        # 2 residual blocks, 256
        channels = 256
        with tf.variable_scope("residual_block_1"):
            x = residual_block_reduce_size(x, channels, is_training)
        with tf.variable_scope("residual_block_2"):
            x = residual_block(x, channels, is_training)
    #x is now Nonex7x7x256
    with tf.variable_scope("yolo_layers"):
        B = 2
        C = 20
        x = x = conv_wrapper(x, shape = [1,1,256,B*5+C], strides = [1, 1, 1, 1], padding = "VALID")
    return x

def image_detection_op(yolo_tensor):
    #input: 1 yolo_tensor size None x 7 x 7 x 30:      P, X, Y, WIDTH, HEIGHT, P, X, Y, WIDTH, HEIGHT, 20C

    #output: None*(0-98 bndbox + class prediction)
    pass



def loss_op(yolo_tensor, gt): #size is None x 7 x 7 x 30, P, X, Y, WIDTH, HEIGHT, P, X, Y, WIDTH, HEIGHT, 20C
    #1st: match gt-label with appropriate grid cell
    batch_size = 1
    S = 7
    gt = (174/486, 101/500, 349/486, 351/500) #xmin ymin xmax ymax
    xmin, ymin, xmax, ymax = gt

    #cell index
    cell_y = int((ymin + ymax)/2 * S)
    cell_x = int((xmin + xmax)/2 * S)

    bndbox1 = yolo_tensor[0, cell_y, cell_x, 1:5] # [x, y, width, height]
    bndbox1_confidence = yolo_tensor[0, cell_y, cell_x, 0]
    bndbox2 = yolo_tensor[0, cell_y, cell_x, 6:10]
    bndbox2_confidence = yolo_tensor[0, cell_y, cell_x, 5]

    bndbox1_iou = iou(box1 = (bndbox1[0], bndbox1[1], bndbox1[2]+bndbox1[0], bndbox1[3]+bndbox1[1]),
                      box2 = gt)

    bndbox2_iou = iou(box1 = (bndbox2[0], bndbox2[1], bndbox2[2]+bndbox2[0], bndbox2[3]+bndbox2[1]),
                      box2 = gt)

    box_index = tf.cond(bndbox1_iou > bndbox2_iou, lambda: tf.constant(0), lambda: tf.constant(1))

    #extract data from gt and yolo_tensor
    gt_x = xmin
    gt_y = ymin
    gt_width = xmax-xmin
    gt_height = ymax-ymin
    x = yolo_tensor[:, cell_y, cell_x, box_index*5 + 1]
    y = yolo_tensor[:, cell_y, cell_x, box_index*5 + 2]
    width = yolo_tensor[:, cell_y, cell_x, box_index*5 + 3]
    height = yolo_tensor[:, cell_y, cell_x, box_index*5 + 4]

    #coord losses
    x_loss = tf.reduce_sum(tf.pow(x - gt_x, 2))
    y_loss = tf.reduce_sum(tf.pow(y - gt_y, 2))
    w_loss = tf.reduce_sum(tf.pow(width - gt_width, 2))
    h_loss = tf.reduce_sum(tf.pow(height - gt_height, 2))
    coord_loss = (x_loss + y_loss + w_loss + h_loss) / batch_size

    #confidence loss
    box1_confidence = yolo_tensor[:, :, :, 0:1]
    box2_confidence = yolo_tensor[:, :, :, 5:6]
    #box_conficences = tf.concat([box1_confidence, box2_confidence], 3)
    gt_box_confidences = np.zeros((1,7,7,2))
    gt_box_confidences[:,cell_y,cell_x,box_index] = 1
    confidence_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = gt_box_confidences[:,:,:,0], logits = box1_confidence))
    confidence_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = gt_box_confidences[:,:,:,1], logits = box2_confidence))

    #class loss
    gt_class = np.zeros((1,20)) #kanske (batch_size, num_objects, 20)
    gt_class[0, 0] = 1
    class_prediction = yolo_tensor[0:1, cell_y, cell_x, 10:30]
    class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = gt_class, logits = class_prob))

    loss = coord_loss + confidence_loss + class_loss
    return loss

def iou(box1, box2):
    #input
    #box1: (xmin1, ymin1, xmax1, ymax1)
    #box2: (xmin2, ymin2, xmax2, ymax2)

    #output: float
    xmin, ymin, xmax, ymax = box1

    predxmin, predymin, predxmax, predymax = box2

    x0 = tf.maximum(xmin, predxmin)
    x1 = tf.minimum(xmax, predxmax)
    y0 = tf.maximum(ymin, predymin)
    y1 = tf.minimum(ymax, predymax)

    intersection_area = (x1-x0) * (y1-y0)
    pred_area = (predxmax - predxmin) * (predymax - predymin)
    gt_area = (xmax - xmin) * (ymax - ymin)
    iou = intersection_area / (gt_area + pred_area - intersection_area)

    return iou

def test_op(yolo_tensor, gt): #size is None x 7 x 7 x 30, P, X, Y, WIDTH, HEIGHT, P, X, Y, WIDTH, HEIGHT, 20C
    #1st: match gt-label with appropriate grid cell
    batch_size = 1
    S = 7
    gt = (174/486, 101/500, 349/486, 351/500) #xmin ymin xmax ymax
    xmin, ymin, xmax, ymax = gt

    #cell index
    cell_y = int((ymin + ymax)/2 * S)
    cell_x = int((xmin + xmax)/2 * S)

    bndbox1 = yolo_tensor[0, cell_y, cell_x, 1:5] # [x, y, width, height]
    bndbox1_confidence = yolo_tensor[0, cell_y, cell_x, 0]
    bndbox2 = yolo_tensor[0, cell_y, cell_x, 6:10]
    bndbox2_confidence = yolo_tensor[0, cell_y, cell_x, 5]

    bndbox1_iou = iou(box1 = (bndbox1[0], bndbox1[1], bndbox1[2]+bndbox1[0], bndbox1[3]+bndbox1[1]),
                      box2 = gt)

    bndbox2_iou = iou(box1 = (bndbox2[0], bndbox2[1], bndbox2[2]+bndbox2[0], bndbox2[3]+bndbox2[1]),
                      box2 = gt)

    box_index = tf.cond(bndbox1_iou > bndbox2_iou, lambda: tf.constant(0), lambda: tf.constant(1))
    #box_index = tf.cast(box_index, tf.int32)

    #extract data from gt and yolo_tensor
    gt_x = xmin
    gt_y = ymin
    gt_width = xmax-xmin
    gt_height = ymax-ymin
    x = yolo_tensor[:, cell_y, cell_x, box_index*5 + 1]
    y = yolo_tensor[:, cell_y, cell_x, box_index*5 + 2]
    width = yolo_tensor[:, cell_y, cell_x, box_index*5 + 3]
    height = yolo_tensor[:, cell_y, cell_x, box_index*5 + 4]

    #coord losses
    x_loss = tf.reduce_sum(tf.pow(x - gt_x, 2))
    y_loss = tf.reduce_sum(tf.pow(y - gt_y, 2))
    w_loss = tf.reduce_sum(tf.pow(width - gt_width, 2))
    h_loss = tf.reduce_sum(tf.pow(height - gt_height, 2))
    coord_loss = (x_loss + y_loss + w_loss + h_loss) / batch_size

    #confidence loss
    box1_confidence = yolo_tensor[:, :, :, 0:1]
    box2_confidence = yolo_tensor[:, :, :, 5:6]
    #box_conficences = tf.concat([box1_confidence, box2_confidence], 3)

    gt_box_confidences = tf.zeros([1,7,7,2])
    gt_box_confidences[:, cell_y, cell_x, box_index] = 1
    confidence_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = gt_box_confidences[:,:,:,0], logits = box1_confidence))
    confidence_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = gt_box_confidences[:,:,:,1], logits = box2_confidence))

    #class loss
    gt_class = np.zeros((1,20)) #kanske (batch_size, num_objects, 20)
    gt_class[0, 0] = 1
    class_prediction = yolo_tensor[0:1, cell_y, cell_x, 10:30]
    class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = gt_class, logits = class_prob))

    loss = coord_loss + confidence_loss + class_loss
    return loss
