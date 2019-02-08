import tensorflow as tf
import gym
import numpy as np
from PIL import Image

class StateProcessor(object):
    def __init__(self, sess):
        self.sess = sess
    
    def process(self, state):
        grayscale_state = tf.image.rgb_to_grayscale(state)
        resized_state = tf.image.resize_images(state, [84, 84])[:,:,2]
        resized_state = tf.reshape(resized_state, [84,84,1])
        return self.sess.run(resized_state)


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    logs_dir = '/tmp/tensorflow/test/tensorboard_image01'
    state_processor = StateProcessor(sess)
    env = gym.make("Seaquest-v0")
    obs = env.reset()
    pro_obs = obs
    tf.summary.image('row_image', tf.reshape(obs,[-1,210,160,3]), 1)
    pro_obs = state_processor.process(pro_obs)
    tf.summary.image('processed_image', tf.reshape(pro_obs, [-1, 84 ,84 ,1]), 1)
    merged = tf.summary.merge_all()
    summary = sess.run(merged)
    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    summary_writer.add_summary(summary)
    summary_writer.close()
