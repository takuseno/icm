import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def _make_convolutional_layers(convs, inpt, with_fc=True, reuse=None):
    with tf.variable_scope('convnet', reuse=reuse):
        out = inpt
        for num_outputs, kernel_size, stride, padding in convs:
            out = layers.convolution2d(out,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='SAME',
                    activation_fn=tf.nn.elu)
        conv_out = layers.flatten(out)
        if with_fc:
            conv_out = layers.fully_connected(conv_out, 256, activation_fn=tf.nn.elu)
    return conv_out

def _make_network(convs, inpt, rnn_state_tuple, num_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        conv_out = _make_convolutional_layers(convs, inpt)

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            rnn_in = tf.expand_dims(conv_out, [0])
            step_size = tf.shape(inpt)[:1]
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=rnn_state_tuple,
                    sequence_length=step_size, time_major=False)
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        policy = layers.fully_connected(rnn_out,
                num_actions, activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None)

        value = layers.fully_connected(rnn_out, 1, activation_fn=None,
                weights_initializer=normalized_columns_initializer(), biases_initializer=None)

    return policy, value, lstm_state

def make_network(convs):
    return lambda *args, **kwargs: _make_network(convs, *args, **kwargs)

def _make_icm(convs, inpt0, inpt1, action, num_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # inverse model
        cnn_out0 = _make_convolutional_layers(convs, inpt0, with_fc=False)
        cnn_out1 = _make_convolutional_layers(convs, inpt1, with_fc=False, reuse=True)

        g = tf.concat([cnn_out0, cnn_out1], 1)
        g = layers.fully_connected(g, 256, activation_fn=tf.nn.elu)

        predicted_action = layers.fully_connected(g, num_actions, activation_fn=tf.nn.softmax)

        action_one_hot = tf.one_hot(action, num_actions, dtype=tf.float32)
        f = tf.concat([cnn_out0, action_one_hot], 1)
        f = layers.fully_connected(f, 256, activation_fn=tf.nn.elu)
        predicted_state = layers.fully_connected(f, 288, activation_fn=None)

    return cnn_out1, predicted_action, predicted_state

def make_icm(convs):
    return lambda *args, **kwargs: _make_icm(convs, *args, **kwargs)
