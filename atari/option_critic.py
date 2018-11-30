import numpy as np
import tensorflow as tf
from network_utils import *
import experience_replay as er

class OptionCritic(object):
    '''
    # TODO
    Target Networkの実装
    ミニバッチ更新の実装
    Intra-option Policyの更新
    Termination functionの更新
    観測データの前処理
    強化学習ループの実装
    Frame skipping
    reward scalingの実装
    '''
    def __init__(self, n_options, n_intra_actions, history_length, batch_size, gamma, learning_rate):
        self.n_options = n_options
        self.n_intra_actions = n_intra_actions
        self.history_lengh = history_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate=learning_rate
        self.sess = tf.Session()
        self.experience_replay = er.Experience_Replay()
        self.build_network()

    def build_network(self):
        # 入力層
        self.input_layer = tf.placeholder(tf.float32, shape=[None, 84, 84, self.history_lengh])
        
        # 畳み込み1
        W_conv1 = weight_variable([8,8,self.history_lengh,32])
        b_conv1 = bias_variable([32])
        l_conv1 = tf.nn.relu(conv2d(self.input_layer, W_conv1,stride=4) + b_conv1) 
        
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
        self.policy_over_options = tf.matmul(h_fc1, W_output1) + b_output1
        
        ## termination function
        W_output2 = weight_variable([512, self.n_options])
        b_output2 = bias_variable([self.n_options])
        self.termination_fuction = tf.nn.sigmoid(tf.matmul(h_fc1, W_output2) + b_output2)
        
        ## intra-option policy
        W_output3 = weight_variable([512, self.n_options*self.n_intra_actions])
        b_output3 = bias_variable([self.n_options*self.n_intra_actions])
        l_out3= tf.nn.softmax(tf.matmul(h_fc1, W_output3) + b_output3)
        self.intra_option_policy = tf.reshape(l_out3, [self.n_options, self.n_intra_actions])
    
    def update_q(self):
        # Policy over optionsの更新，Experience replyを用いる、ミニバッチ学習
        mini_batch = self.experience_replay.get_experience(self.batch_size)
        #for s,a,r,s_,t in mini_batch:

        # for s, a, r, s_, t in mini_batch:
        #     if t:
        #         y = r
        #     else:
        #         max_q = self.get_max_q(s)
        #         y = r + self.gamma * max_q
        #     opt = tf.train.RMSPropOptimizer(self.learning_rate)
        #     train_step = opt.minimize(tf.pow((y - self.get_q(s_, a)),2))
        #     self.sess.run(train_step)
    
    def update_intra_option_policy(self):
        # 誤差の計算式の実装、オンライン学習
        return

    def update_terminal_function(self):
        # TODO 
        # アドバンテージ関数の算出、誤差の計算式の実装、オンライン学習
        return
    
        
    def get_max_q(self, obs):
        return tf.math.max(self.policy_over_options)
    
    def get_q(self, obs, a):
        q_values = self.sess.run(self.policy_over_options, feed_dict={self.input_layer:obs})
        return q_values[a]

    def get_term_prob(self, obs, omega):
        return self.sess.run(self.termination_fuction, feed_dict={self.input_layer: obs})[omega]

    def take_option(self, observation):
        
        return
    
    
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def take_intra_option(self):
        return

    def update_network(self):
        return 
    
    def get_terminate_value(self):
        return 

if __name__ == "__main__":
    OptionCritic(8,18,4)