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
        with tf.variable_scope("bndboxes"):
            yolo_tensor = conv_wrapper(x, shape = [1,1,256,B*5], strides = [1, 1, 1, 1], padding = "VALID")
            yolo_tensor[:, :, :, 0] = (tf.nn.tanh(yolo_tensor[:, :, :, 0]) + 1) / 2
            yolo_tensor[:, :, :, 5] = (tf.nn.tanh(yolo_tensor[:, :, :, 5]) + 1) / 2
            FIX THIS SHIT, DIVIDE UP INTO 3 THINGS, box_tensor, confidence_tensor, class_tensor
        with tf.variable_scope("classes"):
            classes = conv_wrapper(x, shape = [1, 1, 256, C], strides = [1, 1, 1, 1], padding = "VALID")
            class_tensor = tf.nn.softmax(classes)
    return yolo_tensor, class_tensor

def iou(box1, box2):
    #input
    #box1: (xmin1, ymin1, xmax1, ymax1)
    #box2: (xmin2, ymin2, xmax2, ymax2)

    #output: float
    xmin = box1[0]
    ymin = box1[1]
    xmax = box1[2]
    ymax = box1[3]

    predxmin = box2[0]
    predymin = box2[1]
    predxmax = box2[2]
    predymax = box2[3]

    x0 = tf.maximum(xmin, predxmin)
    x1 = tf.minimum(xmax, predxmax)
    y0 = tf.maximum(ymin, predymin)
    y1 = tf.minimum(ymax, predymax)

    intersection_area = (x1-x0) * (y1-y0)
    pred_area = (predxmax - predxmin) * (predymax - predymin)
    gt_area = (xmax - xmin) * (ymax - ymin)
    iou = intersection_area / (gt_area + pred_area - intersection_area)

    return iou

def condition1(batch_count, batch_size, loss, num_objects, yolo_tensor, class_tensor, gt):
    return batch_count < batch_size

def body1(batch_count, batch_size, loss, num_objects, yolo_tensor, class_tensor, gt):

    #while loop
    obj_idx = tf.constant(0)
    result = tf.while_loop(condition2, body2, [batch_count, obj_idx, num_objects[batch_count], loss, yolo_tensor, class_tensor, gt], swap_memory=True)

    batch_loss = result[3]

    loss += batch_loss

    #iterate
    batch_count += 1
    return batch_count, batch_size, loss, num_objects, yolo_tensor, class_tensor, gt

def condition2(batch_count, obj_idx, num_objects, loss, yolo_tensor, class_tensor, gt):
    return obj_idx < num_objects

def body2(batch_count, obj_idx, num_objects, loss, yolo_tensor, class_tensor, gt):
    #do shit

    alpha_coord = 10.0
    alpha_obj_confidence = 2
    alpha_class = 1.0

    gt_box = gt[batch_count, obj_idx, 0:4]
    gt_class_idx = tf.cast(gt[batch_count, obj_idx, 4], tf.int32)
    xmin = gt_box[0]
    ymin = gt_box[1]
    xmax = gt_box[2]
    ymax = gt_box[3]
    S = tf.constant(7.0)

    cell_y = tf.cast(tf.floor((ymin + ymax)/2 * S), tf.int32)
    cell_x = tf.cast(tf.floor((xmin + xmax)/2 * S), tf.int32)

    box0 = yolo_tensor[batch_count, cell_y, cell_x, 1:5] # [x, y, width, height]
    confidence0 = yolo_tensor[batch_count, cell_y, cell_x, 0]
    box1 = yolo_tensor[batch_count, cell_y, cell_x, 6:10]
    confidence1 = yolo_tensor[batch_count, cell_y, cell_x, 5]

    bndbox1_iou = iou(box1 = (box0[0], box0[1], box0[2]+box0[0], box0[3]+box0[1]),
                      box2 = gt_box)

    bndbox2_iou = iou(box1 = (box1[0], box1[1], box1[2]+box1[0], box1[3]+box1[1]),
                      box2 = gt_box)

    box_index = tf.cond(bndbox1_iou > bndbox2_iou, lambda: tf.constant(0), lambda: tf.constant(1))
    box_index = tf.cast(box_index, tf.int32)

    #extract data from gt and yolo_tensor
    gt_x = xmin
    gt_y = ymin
    gt_width = xmax-xmin
    gt_height = ymax-ymin
    x = yolo_tensor[batch_count, cell_y, cell_x, box_index*5 + 1]
    y = yolo_tensor[batch_count, cell_y, cell_x, box_index*5 + 2]
    width = yolo_tensor[batch_count, cell_y, cell_x, box_index*5 + 3]
    height = yolo_tensor[batch_count, cell_y, cell_x, box_index*5 + 4]

    #coord losses
    x_loss = tf.reduce_sum(tf.pow(x - gt_x, 2))
    y_loss = tf.reduce_sum(tf.pow(y - gt_y, 2))
    w_loss = tf.reduce_sum(tf.pow(width - gt_width, 2))
    h_loss = tf.reduce_sum(tf.pow(height - gt_height, 2))
    coord_loss = x_loss + y_loss + w_loss + h_loss

    #confidence loss
    box1_confidence = yolo_tensor[batch_count, :, :, 0]
    box2_confidence = yolo_tensor[batch_count, :, :, 5]
    box_conficences = tf.stack([box1_confidence, box2_confidence])

    one_hot0 = tf.one_hot(indices = box_index,
                          depth = 2,
                          on_value = cell_x,
                          off_value = -1,
                          axis = 0)
    one_hot1 = tf.one_hot(indices = one_hot0,
                          depth = 7,
                          on_value = cell_y,
                          off_value = -1,
                          axis = -1)
    gt_box_confidences = tf.one_hot(indices = one_hot1,
                                    depth = 7,
                                    on_value = alpha_obj_confidence,
                                    off_value = 0,
                                    axis = -1)

    #if something is wrong then cell_x and cell_y is flipped
    confidence_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = gt_box_confidences, logits = box_conficences))

    #class loss
    gt_prob = tf.constant(1.0)
    pred_prob = class_tensor[batch_count, cell_y, cell_x, gt_class_idx]
    class_loss = tf.reduce_sum(- gt_prob * tf.log(pred_prob))

    totalloss = alpha_coord * coord_loss + confidence_loss + alpha_class* class_loss
    loss += totalloss

    #iterate
    obj_idx += 1
    return batch_count, obj_idx, num_objects, loss, yolo_tensor, class_tensor, gt

def loss(yolo_tensor, class_tensor, gt, num_objects, batch_size):
    #yolo_tensor is None x 7 x 7 x 10       ### P, X, Y, WIDTH, HEIGHT, P, X, Y, WIDTH, HEIGHT
    #class_tensor is None x 7 x 7 x 20      ### 20C
    #gt is (batch_size, num_objects, 5)     ### xmin, ymin, xmax, ymax, class prediction index
    #num_objects is [batch_size]

    #1st: match gt-label with appropriate grid cell
    #batch_size = num_objects.get_shape().as_list()[0]

    loss = tf.constant(0.0)
    batch_count = tf.constant(0)
    while_results = tf.while_loop(condition1, body1, [batch_count, batch_size, loss, num_objects, yolo_tensor, class_tensor, gt], swap_memory=True)

    loss = while_results[2]
    loss = loss / tf.cast(batch_size, tf.float32)
    return loss

def detect_objects(yolo_tensor, class_tensor):
    #yolo_tensor is None x 7 x 7 x 10       ### P, X, Y, WIDTH, HEIGHT, P, X, Y, WIDTH, HEIGHT
    #class_tensor is None x 7 x 7 x 20      ### 20C

    box1_confidence = yolo_tensor[0, :, :, 0]
    box2_confidence = yolo_tensor[0, :, :, 5]
    box_confidences = tf.stack([box1_confidence, box2_confidence]) #2, 7, 7

    class_confidences = tf.reduce_max(class_tensor[0,:,:,:], axis = 2)
    class_conf = tf.stack([class_confidences, class_confidences])

    probs = box_confidences * class_conf
    classes = tf.argmax(class_tensor[0,:,:,:], axis = 2)

    box1 = yolo_tensor[0, :, :, 1:5]
    box2 = yolo_tensor[0, :, :, 6:10]
    boxes = tf.stack([box1, box2]) #2, 7, 7
    return boxes, classes, probs

def process_boxes(threshold, boxes, classes, probs):
    final_boxes = []
    final_classes = []
    final_probs = []
    for i in range(7):
        for j in range(7):
            for b in range(2):
                if probs[b,i,j] > threshold:
                    final_boxes.append(boxes[b,i,j])
                    final_classes.append(classes[i,j])
                    final_probs.append(probs[b,i,j])
    return final_boxes, final_classes, final_probs
