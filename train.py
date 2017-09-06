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
    parser.add_argument('--threads', type=int, default=8)
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
    master = Agent(model, icm_model, len(actions), name='global')

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    saver = tf.train.Saver(global_vars)
    if args.load:
        saver.restore(sess, args.load)

    workers = []
    for i in range(args.threads):
        render = False
        if args.render and i == 0:
            render = True
        worker = Worker('worker{}'.format(i), model, icm_model, global_step, env_name, render=render)
        workers.append(worker)

    summary_writer = tf.summary.FileWriter('log', sess.graph)

    if args.render:
        sample_worker = workers.pop(0)

    initialize()

    coord = tf.train.Coordinator()
    threads = []
    for i in range(len(workers)):
        worker_thread = lambda: workers[i].run(sess, summary_writer, saver)
        thread = threading.Thread(target=worker_thread)
        thread.start()
        threads.append(thread)
        time.sleep(0.1)

    if args.render:
        sample_worker.run(sess, summary_writer)

    coord.join(threads)

if __name__ == '__main__':
    main()
