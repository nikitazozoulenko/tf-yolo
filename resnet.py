import tensorflow as tf

EPSILON = 0.00001

def conv_wrapper(x, shape, strides, padding):
    weights = tf.get_variable("weights",
                              shape,
                              initializer = tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases",
                             [shape[3]],
                             initializer = tf.constant_initializer(0.1))

    #tf.histogram_summary("weights_summary", weights)

    conv = tf.nn.conv2d(x,
                        weights,
                        strides = strides,
                        padding = padding)
    return conv + biases

def bn_wrapper(x, is_training):
    gamma = tf.get_variable("gamma",
                            [x.get_shape()[-1]],
                            initializer = tf.constant_initializer(1.0))
    beta = tf.get_variable("beta",
                            [x.get_shape()[-1]],
                            initializer = tf.constant_initializer(1.0))
    moving_mean = tf.get_variable("moving_mean",
                                  [x.get_shape()[-1]],
                                  initializer = tf.constant_initializer(0.0),
                                  trainable = False)
    moving_variance = tf.get_variable("moving_variance",
                                      [x.get_shape()[-1]],
                                      initializer = tf.constant_initializer(1.0),
                                      trainable = False)
    return tf.cond(is_training,
                   lambda: bn_train_time(x, beta, gamma, moving_mean, moving_variance),
                   lambda: bn_test_time(x, beta, gamma, moving_mean, moving_variance))

def bn_train_time(x, beta, gamma, moving_mean, moving_variance):
    mean, variance = tf.nn.moments(x, axes = [0,1,2])
    ALPHA = 0.90
    op_moving_mean = tf.assign(moving_mean,
                               moving_mean * ALPHA + mean * (1-ALPHA))
    op_moving_variance = tf.assign(moving_variance,
                                   moving_variance * ALPHA + variance * (1-ALPHA))
    with tf.control_dependencies([op_moving_mean, op_moving_variance]):
        return tf.nn.batch_normalization(x,
                                         mean,
                                         variance,
                                         offset = beta,
                                         scale = gamma,
                                         variance_epsilon = EPSILON)

def bn_test_time(x, beta, gamma, moving_mean, moving_variance):
    return tf.nn.batch_normalization(x,
                                     moving_mean,
                                     moving_variance,
                                     offset = beta,
                                     scale = gamma,
                                     variance_epsilon = EPSILON)

def residual_block(x, C, is_training):
    with tf.variable_scope("h1_conv_bn"):
        conv1 = conv_wrapper(x, shape = [3,3,C,C], strides = [1, 1, 1, 1], padding = "SAME")
        bn1 = bn_wrapper(conv1, is_training)
    relu1 = tf.nn.relu(bn1)
    with tf.variable_scope("h2_conv_bn"):
        conv2 = conv_wrapper(relu1, shape = [3,3,C,C], strides = [1, 1, 1, 1], padding = "SAME")
        bn2 = bn_wrapper(conv2, is_training)

    res = x + bn2
    return tf.nn.relu(res)

def residual_block_reduce_size(x, C, is_training):
    last_C = x.get_shape().as_list()[-1]
    with tf.variable_scope("h1_conv_bn"):
        conv1 = conv_wrapper(x, shape = [3,3,last_C,C], strides = [1, 2, 2, 1], padding = "VALID")
        bn1 = bn_wrapper(conv1, is_training)
    relu1 = tf.nn.relu(bn1)
    with tf.variable_scope("h2_conv_bn"):
        conv2 = conv_wrapper(relu1, shape = [3,3,C,C], strides = [1, 1, 1, 1], padding = "SAME")
        bn2 = bn_wrapper(conv2, is_training)

    return tf.nn.relu(bn2)
