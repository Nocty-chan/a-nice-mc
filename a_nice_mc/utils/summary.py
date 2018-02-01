import tensorflow as tf

def variable_summaries(name, var):
    with tf.name_scope(name):
        tf.summary.scalar('value', var)

def variable_stats(name, var):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        meansum = tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        stddevsum = tf.summary.scalar('stddev', stddev)
        maxsum = tf.summary.scalar('max', tf.reduce_max(var))
        minsum = tf.summary.scalar('min', tf.reduce_min(var))
        return tf.summary.merge([meansum, stddevsum, maxsum, minsum])
