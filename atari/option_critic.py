import numpy as np
import tensorflow as tf
from network_utils import *

class OptionCritic(object):
    def __init__(self, n_options, n_intra_actions, history_length):
        self.n_options = n_options
        self.n_intra_actions = n_intra_actions
        self.history_lengh = history_length
        self.build_network()

    def build_network(self):
        # 入力層
        input = tf.placeholder(tf.float32, shape=[None, 84, 84, self.history_lengh])
        
        # 畳み込み1
        W_conv1 = weight_variable([8,8,self.history_lengh,32])
        b_conv1 = bias_variable([32])
        l_conv1 = tf.nn.relu(conv2d(input, W_conv1,stride=4) + b_conv1) 
        
        # 畳み込み2
        W_conv2 = weight_variable([4,4,32,64])
        b_conv2 = bias_variable([64])
        l_conv2 = tf.nn.relu(conv2d(l_conv1, W_conv2, stride=2)+b_conv2)
        
        # 畳み込み3
        W_conv3 = weight_variable([3,3,64,64])
        b_conv3 = bias_variable([64])
        l_conv3 = tf.nn.relu(conv2d(l_conv2, W_conv3, stride=1)+b_conv3)

        # 全結合層1
        W_fc1 = weight_variable([3 * 3 * 64, 512])
        b_fc1 = weight_variable([512])
        h_flat = tf.reshape(l_conv3, [-1, 3 * 3 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        # 出力層
        ## policy over options
        W_output1 = weight_variable([512, self.n_options])
        b_output1 = bias_variable([self.n_options])
        self.policy_over_options = tf.nn.softmax(tf.matmul(h_fc1, W_output1) + b_output1)
        
        ## termination function
        W_output2 = weight_variable([512, self.n_options])
        b_output2 = bias_variable([self.n_options])
        self.termination_fuction = tf.nn.softmax(tf.matmul(h_fc1, W_output2) + b_output2)
        
        ## intra-option policy
        W_output3 = weight_variable([512, self.n_options*self.n_intra_actions])
        b_output3 = bias_variable([self.n_options*self.n_intra_actions])
        l_out3= tf.nn.softmax(tf.matmul(h_fc1, W_output3) + b_output3)
        self.intra_option_policy = tf.reshape(l_out3, [self.n_options, self.n_intra_actions])

    def take_option(self):
        return
    
    def take_intra_option(self):
        return

    def update_network(self):
        return 
    
    def get_terminate_value(self):
        return 

if __name__ == "__main__":
    OptionCritic(8,18,4)