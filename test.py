from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from yolo_net import *
from scipy.ndimage.interpolation import zoom
from PIL import Image

def func():
    box_index = 0
    cell_x = 3
    cell_y = 5
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
    x =                 tf.one_hot(indices = one_hot1,
                                    depth = 7,
                                    on_value = 1,
                                    off_value = 0,
                                    axis = -1)
    return tf.cast(x > 0, tf.float32)

operation = func()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(operation)
    print(output)







0.20799375  0.27756768  0.54199749  0.37365949
0.26599979  0.31315663  0.12800831  0.12455814
0.05199394  0.6725949   0.03599799  0.17438549
0.38999274  0.64057106  0.03600428  0.17438677

0.26600832  0.31292093  0.12784322  0.12518309
0.20837656  0.27778473  0.5414362   0.37377453
0.38986728  0.64070261  0.03557985  0.17473495
0.05194731  0.67241746  0.03598529  0.17457847
