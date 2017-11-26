from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from resnet import *

def YOLO_network(x, is_training):
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
        yolo_tensor = conv_wrapper(x, shape = [1,1,256,B*5 + C], strides = [1, 1, 1, 1], padding = "VALID")

        box_tensor = yolo_tensor[:,:,:,0:8]

        confidence_tensor = yolo_tensor[:,:,:,8:10]
        #confidence_tensor = (tf.nn.tanh(confidence_tensor)+1)/2

        class_tensor = yolo_tensor[:,:,:,10:30]
        class_tensor = tf.nn.softmax(class_tensor)
    return box_tensor, confidence_tensor, class_tensor

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

def condition1(batch_count, batch_size, loss, num_objects, box_tensor, confidence_tensor, class_tensor, gt):
    return batch_count < batch_size

def body1(batch_count, batch_size, loss, num_objects, box_tensor, confidence_tensor, class_tensor, gt):

    #while loop
    obj_idx = tf.constant(0)
    gt_box_confidences = tf.zeros([2,7,7])
    batch_coord_loss = tf.constant(0.0)
    batch_confidence_loss = tf.constant(0.0)
    batch_class_loss = tf.constant(0.0)

    result = tf.while_loop(condition2, body2, [batch_count, obj_idx, num_objects[batch_count], [batch_coord_loss, batch_confidence_loss, batch_class_loss], box_tensor, confidence_tensor, class_tensor, gt, gt_box_confidences], swap_memory=True)
    batch_loss = result[3]

    batch_coord_loss = batch_loss[0]
    batch_class_loss = batch_loss[2]

    #confidence loss
    box1_confidence = confidence_tensor[batch_count, :, :, 0]
    box2_confidence = confidence_tensor[batch_count, :, :, 1]
    box_conficences = tf.stack([box1_confidence, box2_confidence])

    gt_box_confidences = result[8]
    gt_box_confidences = tf.minimum(gt_box_confidences, tf.ones([2,7,7]))
    mask = tf.cast(gt_box_confidences > 0, tf.float32)
    batch_confidence_loss = tf.pow(box_conficences - gt_box_confidences, 2)
    #KANSKE INTE BEHÖVER EN SÅ JOBBIG CONF_LOSS FUNKTION, EXAKT SOM CLASS_LOSS

    alpha_obj_confidence = 5.0
    batch_confidence_loss += mask * alpha_obj_confidence * batch_confidence_loss

    coord_loss = loss[0] + batch_coord_loss
    confidence_loss = loss[1] + tf.reduce_sum(batch_confidence_loss)
    class_loss = loss[2] + batch_class_loss

    #iterate
    batch_count += 1
    return batch_count, batch_size, [coord_loss, confidence_loss, class_loss], num_objects, box_tensor, confidence_tensor, class_tensor, gt

def condition2(batch_count, obj_idx, num_objects, loss, box_tensor, confidence_tensor, class_tensor, gt, gt_box_confidences):
    return obj_idx < num_objects

def body2(batch_count, obj_idx, num_objects, loss, box_tensor, confidence_tensor, class_tensor, gt, gt_box_confidences):
    #do shit
    coord_loss = loss[0]
    confidence_loss = loss[1]
    class_loss = loss[2]

    S = tf.constant(7.0)

    gt_box = gt[batch_count, obj_idx, 0:4]
    gt_class_idx = tf.cast(gt[batch_count, obj_idx, 4], tf.int32)
    xmin = gt_box[0]
    ymin = gt_box[1]
    xmax = gt_box[2]
    ymax = gt_box[3]

    cell_y = tf.cast(tf.floor((ymin + ymax)/2 * S), tf.int32)
    cell_x = tf.cast(tf.floor((xmin + xmax)/2 * S), tf.int32)

    box0 = box_tensor[batch_count, cell_y, cell_x, 0:4] # [x, y, width, height]
    box1 = box_tensor[batch_count, cell_y, cell_x, 4:8]

    bndbox1_iou = iou(box1 = (box0[0], box0[1], box0[2]+box0[0], box0[3]+box0[1]),
                      box2 = gt_box)

    bndbox2_iou = iou(box1 = (box1[0], box1[1], box1[2]+box1[0], box1[3]+box1[1]),
                      box2 = gt_box)

    box_index = tf.cond(bndbox1_iou > bndbox2_iou, lambda: tf.constant(0), lambda: tf.constant(1))
    box_index = tf.cast(box_index, tf.int32)

    #extract data from gt and box_tensor
    gt_x = xmin
    gt_y = ymin
    gt_width = xmax-xmin
    gt_height = ymax-ymin
    x = box_tensor[batch_count, cell_y, cell_x, box_index*4]
    y = box_tensor[batch_count, cell_y, cell_x, box_index*4 + 1]
    width = box_tensor[batch_count, cell_y, cell_x, box_index*4 + 2]
    height = box_tensor[batch_count, cell_y, cell_x, box_index*4 + 3]

    #coord losses
    x_loss = tf.reduce_sum(tf.pow(x - gt_x, 2))
    y_loss = tf.reduce_sum(tf.pow(y - gt_y, 2))
    w_loss = tf.reduce_sum(tf.pow(width - gt_width, 2))
    h_loss = tf.reduce_sum(tf.pow(height - gt_height, 2))
    coord_loss += (x_loss + y_loss + w_loss + h_loss)

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
    gt_box_confidences += tf.one_hot(indices = one_hot1,
                                    depth = 7,
                                    on_value = 1.0,
                                    off_value = 0.0,
                                    axis = -1)
    #if something is wrong then cell_x and cell_y is flipped


    #class loss
    gt_prob = tf.constant(1.0)
    pred_prob = class_tensor[batch_count, cell_y, cell_x, gt_class_idx]
    class_loss += tf.reduce_sum(- gt_prob * tf.log(pred_prob))

    #iterate
    obj_idx += 1
    return batch_count, obj_idx, num_objects, [coord_loss, confidence_loss, class_loss], box_tensor, confidence_tensor, class_tensor, gt, gt_box_confidences

def loss(box_tensor, confidence_tensor, class_tensor, gt, num_objects, batch_size):
    #yolo_tensor is batch_size x 7 x 7 x 10       ### P, X, Y, WIDTH, HEIGHT, P, X, Y, WIDTH, HEIGHT
    #class_tensor is batch_size x 7 x 7 x 20      ### 20C
    #gt is (batch_size, num_objects, 5)     ### xmin, ymin, xmax, ymax, class prediction index
    #num_objects is [batch_size]

    #1st: match gt-label with appropriate grid cell
    #batch_size = num_objects.get_shape().as_list()[0]

    coord_loss = tf.constant(0.0)
    confidence_loss = tf.constant(0.0)
    class_loss = tf.constant(0.0)

    batch_count = tf.constant(0)
    while_results = tf.while_loop(condition1, body1, [batch_count, batch_size, [coord_loss, confidence_loss, class_loss],
                                    num_objects, box_tensor, confidence_tensor, class_tensor, gt], swap_memory=True)

    loss = while_results[2]

    alpha_coord = 3.0
    alpha_class = 1.0


    coord_loss = loss[0] * alpha_coord / tf.cast(batch_size, tf.float32)
    confidence_loss = 1000.0 * loss[1] / tf.cast(batch_size, tf.float32)
    class_loss = loss[2] * alpha_class / tf.cast(batch_size, tf.float32)
    tf.summary.scalar("coord_loss", coord_loss)
    tf.summary.scalar("confidence_loss", confidence_loss)
    tf.summary.scalar("class_loss", class_loss)

    total_loss = coord_loss + confidence_loss + class_loss
    tf.summary.scalar("coord_loss", coord_loss)

    return total_loss

def detect_objects(box_tensor, confidence_tensor, class_tensor):
    #yolo_tensor is None x 7 x 7 x 10       ### P, X, Y, WIDTH, HEIGHT, P, X, Y, WIDTH, HEIGHT
    #class_tensor is None x 7 x 7 x 20      ### 20C

    box1_confidence = confidence_tensor[0, :, :, 0]
    box2_confidence = confidence_tensor[0, :, :, 1]
    box_confidences = tf.stack([box1_confidence, box2_confidence]) #2, 7, 7

    class_confidences = tf.reduce_max(class_tensor[0,:,:,:], axis = 2)
    class_conf = tf.stack([class_confidences, class_confidences])

    probs = box_confidences * class_conf
    classes = tf.argmax(class_tensor[0,:,:,:], axis = 2)

    box1 = box_tensor[0, :, :, 0:4]
    box2 = box_tensor[0, :, :, 4:8]
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
