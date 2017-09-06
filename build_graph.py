import numpy as np
import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(model, icm_model, num_actions, optimizer, scope='a3c', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_input0 = tf.placeholder(tf.float32, [None, 42, 42, 4], name='obs0')
        obs_input1 = tf.placeholder(tf.float32, [None, 42, 42, 4], name='obs1')
        actions_ph = tf.placeholder(tf.int32, [None], name='action')
        rnn_state_ph0 = tf.placeholder(tf.float32, [1, 256])
        rnn_state_ph1 = tf.placeholder(tf.float32, [1, 256])

        # ICM
        encoded_obs1, predicted_action, predicted_state = icm_model(
                obs_input0, obs_input1, actions_ph, num_actions, scope='icm')
        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        inverse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=predicted_action, labels=actions_ph), name='inverse_loss')
        forward_loss = 0.5 * tf.reduce_mean(
                tf.square(tf.subtract(predicted_state, encoded_obs1)), name='forward_loss')
        icm_loss = 20.0 * (0.8 * inverse_loss + 0.2 * forward_loss)

        # A3C
        target_values_ph = tf.placeholder(tf.float32, [None], name='value')
        advantages_ph = tf.placeholder(tf.float32, [None], name='advantage')
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(rnn_state_ph0, rnn_state_ph1)
        policy, value, state_out = model(obs_input0, rnn_state_tuple, num_actions, scope='a3c')

        actions_one_hot = tf.one_hot(actions_ph, num_actions, dtype=tf.float32)
        responsible_outputs = tf.reduce_sum(policy * actions_one_hot, [1])
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        value_loss = tf.nn.l2_loss(target_values_ph - tf.reshape(value, [-1]))
        entropy = -tf.reduce_sum(policy * log_policy)
        policy_loss = -tf.reduce_sum(tf.reduce_sum(
                tf.multiply(log_policy, actions_one_hot)) * advantages_ph + entropy * 0.01)

        a3c_loss = 0.5 * value_loss + policy_loss
        loss_summary = tf.summary.scalar('{}_loss'.format(scope), a3c_loss)

        icm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/icm'.format(scope))
        icm_gradients, _ = tf.clip_by_global_norm(tf.gradients(icm_loss, icm_vars), 40.0)
        a3c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/a3c'.format(scope))
        a3c_gradients, _ = tf.clip_by_global_norm(tf.gradients(a3c_loss, a3c_vars), 40.0)

        global_icm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/icm')
        icm_optimize_expr = optimizer.apply_gradients(zip(icm_gradients, global_icm_vars))
        global_a3c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/a3c')
        a3c_optimize_expr = optimizer.apply_gradients(zip(a3c_gradients, global_a3c_vars))
        optimize_expr = tf.group(icm_optimize_expr, a3c_optimize_expr)

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        update_local_expr = []
        for local_var, global_var in zip(local_vars, global_vars):
            update_local_expr.append(local_var.assign(global_var))
        update_local_expr = tf.group(*update_local_expr)
        update_local = util.function([], [], updates=[update_local_expr])

        train = util.function(
            inputs=[
                obs_input0, obs_input1, rnn_state_ph0, rnn_state_ph1,
                        actions_ph, target_values_ph, advantages_ph
            ],
            outputs=[loss_summary, a3c_loss],
            updates=[optimize_expr]
        )

        state_value = util.function([obs_input0, rnn_state_ph0, rnn_state_ph1], value)

        act = util.function(inputs=[obs_input0, rnn_state_ph0, rnn_state_ph1], outputs=[policy, state_out])

        bonus = util.function([obs_input0, obs_input1, actions_ph], forward_loss)

    return act, train, update_local, state_value, bonus
