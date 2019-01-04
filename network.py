import tensorflow as tf

def preprocess_tf(x):
    mean, variance = tf.nn.moments(x)

    elms = tf.reduce_prod(x.shape)
    adjusted_stddev = max(variance, 1.0/sqrt(elms))
    ret = (x - mean) / adjusted_stddev
    return ret

def create_network(config, x, scope='network', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('l1', reuse=reuse):
            x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu)
            x = tf.contrib.layers.max_pool2d(x, kernel_size=4, stride=2, padding='SAME', scope='max_pool')

        with tf.variable_scope('conv2', reuse=reuse):
            x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.contrib.layers.max_pool2d(x, kernel_size=2, stride=1, padding='SAME', scope='max_pool')

        with tf.variable_scope('flat', reuse=reuse):
            flat = tf.layers.flatten(x)


        with tf.variable_scope('dense1', reuse=reuse):
            out = tf.layers.dense(inputs=flat, units=config.get('num_dense_units'), use_bias=True, name='layer')

        with tf.variable_scope('output_dense', reuse=reuse):
            out = tf.layers.dense(inputs=out, units=config.get('num_outputs'), use_bias=True, name='layer')

    return out
