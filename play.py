import threading
import multiprocessing
import argparse
import cv2
import gym
import copy
import os
import time
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from actions import get_action_space
from network import make_network, make_icm
from agent import Agent
from worker import Worker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ppaquette/SuperMarioBros-1-1-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--load', type=str)
    args = parser.parse_args()

    sess = tf.Session()
    sess.__enter__()

    model = make_network(
        [[32, 3, 2, 0], [32, 3, 2, 0], [32, 3, 2, 0], [32, 3, 2, 0]])
    icm_model = make_icm(
        [[32, 3, 2, 0], [32, 3, 2, 0], [32, 3, 2, 0], [32, 3, 2, 0]])

    env_name = args.env
    actions = np.arange(14).tolist()

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

    worker = Worker('global', model, icm_model,
            global_step, env_name, render=args.render, training=False)
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

    saver = tf.train.Saver(global_vars)
    if args.load:
        saver.restore(sess, args.load)

    summary_writer = tf.summary.FileWriter('log', sess.graph)

    worker.run(sess, summary_writer, saver)

if __name__ == '__main__':
    main()
