import numpy as np
import tensorflow as tf
from network_utils import *
from replay_buffer import ReplayBuffer

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
    def __init__(self, sess, n_options, n_intra_actions, replay_buffer, history_length, batch_size, gamma, learning_rate, epsilon):
        self.n_options = n_options
        self.n_intra_actions = n_intra_actions
        self.history_lengh = history_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate=learning_rate
        self.sess = sess
        self.replay_buffer = replay_buffer
        self.build_network()
        self.epsilon = epsilon

    def build_network(self):
        # 入力層
        self.input_layer = tf.placeholder(tf.float32, shape=[None, 84, 84, 1])
        
        # 畳み込み1
        W_conv1 = weight_variable([8,8,1,32])
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
        W_fc1 = weight_variable([11 * 11 * 64, 512])
        b_fc1 = weight_variable([512])
        h_flat = tf.reshape(l_conv3, [-1, 11 * 11 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        # 出力層
        ## policy over options
        W_output1 = weight_variable([512, self.n_options])
        b_output1 = bias_variable([self.n_options])
        self.policy_over_options = tf.matmul(h_fc1, W_output1) + b_output1
        self.policy_over_options = tf.reduce_mean(self.policy_over_options, 0)
        
        ## termination function
        W_output2 = weight_variable([512, self.n_options])
        b_output2 = bias_variable([self.n_options])
        self.termination_function = tf.sigmoid(tf.matmul(h_fc1, W_output2) + b_output2)
        self.termination_function = tf.reduce_mean(self.termination_function, 0)
        ## intra-option policy
        W_output3 = weight_variable([512, self.n_options*self.n_intra_actions])
        b_output3 = bias_variable([self.n_options*self.n_intra_actions])
        l_out3= tf.nn.softmax(tf.matmul(h_fc1, W_output3) + b_output3)
        self.intra_option_policy = tf.reshape(l_out3, [-1, self.n_options, self.n_intra_actions])
        self.intra_option_policy = tf.reduce_mean(self.intra_option_policy, 0)
    
    def update_critic(self):
        # Policy over optionsの更新，Experience replyを用いる、ミニバッチ学習
        # Targetは割引累積報酬和
        # TODO minibatch入力
        mini_batch = self.replay_buffer.get_experience(self.batch_size)
        #for s,a,r,s_,t in mini_batch:
        opt = tf.train.RMSPropOptimizer(self.learning_rate)
        total_loss = tf.zeros(shape=[1])
        for s, omega, a, r, s_, t in mini_batch:
            y = self.get_target(s, omega, a, r, t)
            loss = tf.pow((y - self.get_q_u(omega, a)),2) # #actions tensors
            total_loss = tf.Add(total_loss, self.sess.run(loss, feed_dict={self.input_layer: s_}))
        mini_batch_loss = tf.divide(total_loss, self.batch_size) # minibatch_learning
        train_step = opt.minimize(mini_batch_loss)
        init = tf.global_variables_initializer()
        self.sess.run(train_step)

    def update_actor(self, s, omega, a, s_, r):
        # 誤差の計算式の実装、オンライン学習
        # アドバンテージ関数の算出、誤差の計算式の実装、オンライン学習
        
        intra_option_selection = self.get_intra_option_ope(omega, a) # 現状態入力
        q_u_op = r + self.gamma * self.get_utility(omega) # q_uの近似
        terminate_selection = self.get_term_prob(omega)
        a_omega, q_u = self.sess.run([self.get_q_omega(omega) - self.get_max_q_omega(), q_u_op], feed_dict={self.input_layer:s_})
        opt = tf.train.AdamOptimizer()
        intra_option_train_op = opt.minimize(-intra_option_selection * q_u)
        terminate_train_op = opt.minimize(terminate_selection * a_omega)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run([intra_option_train_op, terminate_train_op], feed_dict={self.input_layer:s})
        
    def get_target(self, s, omega, a, r, t):
        loss = r - self.get_q_u(omega, a)
        y = self.sess.run(loss, feed_dict={self.input_layer: s})
        if not t:
            max_q_u = self.sess.run(self.get_max_q_u(), feed_dict={self.input_layer: s})
            # 入力の状態が異なるためにsess.runで値を確定しておく必要がある。
            y = y + self.gamma * ((1 - self.get_term_prob(omega)) * self.get_q_omega(omega) + self.get_term_prob(omega) * max_q_u)
        return y
    
    def get_utility(self, omega):
        v = self.get_max_q_omega() # greedy選択の場合、確率選択なら平均を取る。
        return (1 - self.get_term_prob(omega)) * self.get_q_omega(omega) + self.get_term_prob(omega) * v

    def get_q_omega(self, omega):
        return self.policy_over_options[omega]

    def get_max_q_omega(self):
        q_omega_list = []
        for option in range(self.n_options):
            q_omega_list.append(self.get_q_omega(option))
        return tf.reduce_max(q_omega_list)

    def get_max_q_u(self):
        return tf.math.max(self.policy_over_options)
        
    
    def get_q_u(self, omega, act):
        # 要修正
        return self.policy_over_options[self.n_intra_actions * omega + act]

    def get_term_prob(self, omega):
        return self.termination_function[omega]

    def take_option(self, observation):
        randomize = np.random.random()
        if randomize > self.epsilon:
            return np.random.choice(self.n_options)
        else:
            return np.argmax(self.sess.run(self.policy_over_options, feed_dict={self.input_layer: observation}))
    
    def take_termination(self, state):
        return self.sess.run(self.termination_function, feed_dict={self.input_layer:state})
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + tf.exp(-x))

    def get_intra_option_ope(self, omega, a):
        return self.intra_option_policy[omega][a]

    def take_intra_option(self, state):
        return self.sess.run(self.intra_option_policy, feed_dict={self.input_layer:state})

if __name__ == "__main__":
    # TODO
    sess = tf.Session()
    OptionCritic(sess, 8, 18, 4, 32, 0.9, 0.01)