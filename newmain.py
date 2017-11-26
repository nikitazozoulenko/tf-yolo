from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import threading
import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from yolo_net import *

MAX_NUM_OBJECTS_VOC = 56
batch_size = 2

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

def python_jpeg_function(index, directory, images_filenames, list_length):
    image = Image.open(directory.decode("utf-8") + "/JPEGImages/" + images_filenames[index].decode("utf-8"))
    image = image.resize((259,259))
    image_array = (np.asarray(image) / 255)

    return image_array

def python_xml_function(index, directory, labels_filenames):
    tree = ET.parse(directory.decode("utf-8")+"/Annotations/"+labels_filenames[index].decode("utf-8"))
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    gt_array = np.zeros((MAX_NUM_OBJECTS_VOC, 5))
    counter = 0
    for obj in root.iter("object"):
        if(counter < MAX_NUM_OBJECTS_VOC):
            obj_class = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            gt_array[counter, 0] = xmin / width
            gt_array[counter, 1] = ymin / height
            gt_array[counter, 2] = xmax / width
            gt_array[counter, 3] = ymax / height
            gt_array[counter, 4] = classtoint_dict[obj_class]
            counter += 1
    return gt_array, counter

def python_function(directory, images_filenames, labels_filenames, list_length):
    #randomize which file to read
    index = np.random.randint(0, list_length)
    #read corresponding jpeg and xml
    image_array = python_jpeg_function(index, directory, images_filenames, list_length)
    gt_array, num_objects = python_xml_function(index, directory, labels_filenames)
    image_array = image_array.astype(np.float32)
    gt_array = gt_array.astype(np.float32)
    return [image_array, gt_array, num_objects]
def create_enqueue_op(queue, images_filenames, labels_filenames, list_length, directory):
    #read gt and jpeg with PIL with a python function wrapper
    resized_image_array, gt_array, num_objects = tf.py_func(python_function, [directory, images_filenames, labels_filenames, list_length], [tf.float32, tf.float32, tf.int32])
    return queue.enqueue([resized_image_array, gt_array, [num_objects]])

def create_queue(directory, images_filenames, labels_filenames, list_length, num_threads):
    # create the queue
    queue = tf.FIFOQueue(capacity=1000, shapes = [[259, 259, 3],[MAX_NUM_OBJECTS_VOC, 5],[1]], dtypes=[tf.float32, tf.float32, tf.int32])

    # create our enqueue_op for this queue
    enqueue_op = create_enqueue_op(queue, images_filenames, labels_filenames, list_length, directory)

    # create a QueueRunner and add to queue runner list, probably only need 1 thread
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enqueue_op] * num_threads))
    return queue

def get_next_batch():
    num_threads = 10
    directory = "E:/Datasets/VOC2012"
    # create a list of all our filenames
    images_filenames = os.listdir(directory+"/JPEGImages")
    labels_filenames = os.listdir(directory+"/Annotations")
    list_length = len(images_filenames)
    # create a filename queue
    queue = create_queue(directory, images_filenames, labels_filenames, list_length, num_threads)
    return queue.dequeue_many(batch_size)


is_training = tf.placeholder(tf.bool)
image_array, gt_array, num_objects = get_next_batch()

box_tensor, confidence_tensor, class_tensor = YOLO_network(image_array, is_training)

loss_op = loss(box_tensor, confidence_tensor, class_tensor, gt_array, num_objects, batch_size)
train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_op)

#detection_op = detect_objects(box_tensor, confidence_tensor, class_tensor)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    for _ in range(1):
        sess.run(train_op, feed_dict = {is_training : True})



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
