import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import gym
from gym import wrappers, logger
import gym_pinball
import pdb
import os
from datetime import datetime

class OptionCriticAgent(object):
    """
    価値関数近似：Fourier basis, https://scholarworks.umass.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1100&context=cs_faculty_pubs

    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.basis_order = 3
        self.shape_state = observation_space.shape
        self.n_options = 4
        self.options = [Option(action_space.n, observation_space.shape[0]) for i in range(self.n_options)]
        self.w_q = np.random.rand(len(self.options), action_space.n, self.basis_order) # the number of orders
        self.c_phi = np.random.randint(0, self.basis_order + 1, (len(self.options), action_space.n, self.basis_order, observation_space.shape[0]))
        self.c_phi = self.c_phi.astype(np.float64)
        self.q_u = np.random.rand(len(self.options), action_space.n)
        self.epsilon = 0.01
        self.gamma = 0.99
        self.lr_wq = 0.05
        self.lr_cphi = 0.05
        # variables for analysis
        self.td_error_list = []

    def act(self, observation, o):
        option = self.options[o]
        q_u_list = self._get_q_u_list(observation, o)
        action = np.argmax(option.get_intra_option_dist(q_u_list))
        return action
    
    def update(self, pre_obs, a, obs, r, done, o):
        q_u_list = self._get_q_u_list(pre_obs, o)
        td_error = r - q_u_list[a]
        option = self.options[o]
        if not done:
            term_prob = option.get_terminate(obs)
            q_omega_list = self._get_q_omega_list(obs) #戦犯
            td_error += self.gamma * ((1 - term_prob) * q_omega_list[o] + term_prob * np.max(q_omega_list))
        self.td_error_list.append(abs(td_error))
        self._update_w_q(td_error, a, o, pre_obs)
        self._update_c_phi(td_error, a, o, pre_obs)
        q_omega = self._get_q_omega(obs, o)
        v_omega = self._get_v_omega(obs)
        option.update(a, obs, q_u_list, q_omega, v_omega)

    def _update_w_q(self, td_error, a, o, obs):
        delta_list = []
        for c_phi_a in self.c_phi[o][a]:
            delta = np.cos(np.pi * np.dot(c_phi_a, obs))
            delta_list.append(self.lr_wq * delta * td_error)
        update_list = np.array(delta_list)
        self.w_q[o][a] += update_list
    
    def _update_c_phi(self, td_error, a, o, obs):
        delta_list = []
        for i in range(self.basis_order):
            c_delta_list = []
            basis_delta = self.w_q[o][a][i] * np.sin(np.pi * np.dot(self.c_phi[o][a][i], obs)) * np.pi
            for j in range(len(obs)):
                c_delta = self.lr_cphi * basis_delta * obs[j] * td_error
                c_delta_list.append(c_delta)
            delta_list.append(c_delta_list)
        update_list = np.array(delta_list)
        #if self.c_phi[o].shape[0] == update_list.shape[0] and self.c_phi[o].shape[1] == update_list.shape[1]:
        self.c_phi[o][a] += update_list

    def _get_q_u(self, obs, o, a):
        """
        Fourier basis of order 3
        """
        phis = np.array([self._get_phi(obs, o, a, i) for i in range(self.basis_order)])
        appr_q = np.sum(np.dot(self.w_q[o], phis))
        return appr_q

    def _get_q_u_list(self, obs, o):
        return np.array([self._get_q_u(obs, o, a) for a in range(self.action_space.n)])

    def _get_q_omega(self, obs, o):
        option = self.options[o]
        q_u_list = self._get_q_u_list(obs, o)
        policy = option.get_intra_option_dist(q_u_list)
        return np.dot(policy, q_u_list)

    def _get_q_omega_list(self, obs):
        return [self._get_q_omega(obs, o) for o in range(len(self.options))]

    def _get_v_omega(self, obs):
        """
        Policy over options is decided determistically.
        """
        q_omega_list = self._get_q_omega_list(obs)
        return np.max(q_omega_list)

    def _get_phi(self, obs, o, a, i):
        """
        Basis function: 基底関数
        """
        c = self.c_phi[o][a][i]
        x = np.array(obs)
        in_tri = np.pi * np.dot(c, x)
        return np.cos(in_tri)

    def get_option(self, obs):
        rand = np.random.rand()
        if rand > self.epsilon:
            q_omega_list = self._get_q_omega_list(obs)
            return np.argmax(q_omega_list)
        else:
            return np.random.choice(self.n_options)

    def get_terminate(self, obs, o):
        return self.options[o].get_terminate(obs)


class Option(object):
    def __init__(self, n_actions, n_obs):
        self.n_actions = n_actions
        self.theta = np.random.rand(n_actions)
        self.vartheta = np.random.rand(n_obs)
        self.lr_theta = 0.01
        self.lr_vartheta = 0.01
        # variables for analysis     

    def update(self, a, obs, q_u_list, q_omega, v_omega):
        self._update_theta(a, obs, q_u_list)
        self._update_vartheta(obs, q_omega, v_omega)
    
    def _update_theta(self, a, obs, q_u_list):
        """
        intra option policy gradient theorem
        """
        #delta = -self.theta**-2*(q_u_list[a]+ np.sum(q_u_list))*q_u_list[a]
        delta = -self.theta[a]**-2 * q_u_list[a]**2 * ( 1 - self.get_intra_option_dist(q_u_list)[a])
        self.theta += self.lr_theta * q_u_list[a] * delta

    def _update_vartheta(self, obs, q_omega, v_omega):
        """
        termination function gradient theorem
        """
        advantage = q_omega - v_omega
        beta = self.get_terminate(obs)
        for i, s in enumerate(obs):
            self.vartheta[i] -= self.lr_vartheta * advantage * s * beta * (1 - beta)

    def get_terminate(self, obs):
        """
        linear-sigmoid functions
        """
        linear_sum = np.dot(self.vartheta, obs)
        return 1/(1 + self.exp(linear_sum))

    def get_intra_option_dist(self, q_u_list):
        """
        Boltzmann policies
        """
        energy = q_u_list/self.theta
        numerator = self.exp(energy)
        denominator = np.sum(numerator)
        return numerator/denominator
    
    def exp(self, x):
        np.where(x > 708, 708, x)
        return np.exp(x)
        
def export_csv(file_path, file_name, array):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    array = pd.DataFrame(array)
    saved_path = os.path.join(file_path, file_name)
    array.to_csv(saved_path)

def moved_average(data, window_size):
    b=np.ones(window_size)/window_size
    return np.convolve(data, b, mode='same')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='PinBall-v0', help='Select the environment to run')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    env = gym.make(args.env_id)

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = OptionCriticAgent(env.action_space, env.observation_space)
    episode_count = 10
    reward = 0
    done = False
    total_reward_list = []
    steps_list = []
    try:
        for i in range(episode_count):
            total_reward = 0
            n_steps = 0
            print("episode: {}".format(i))
            ob = env.reset()
            option = agent.get_option(ob)
            while True:
                # if i % 5 == 0:
                #     env.render()
                n_steps += 1
                action = agent.act(ob, option)
                pre_obs = ob
                ob, reward, done, _ = env.step(action)
                total_reward += reward
                if args.debug:
                    agent.update(pre_obs, action, ob, reward, done, option)
                    import pdb; pdb.set_trace()
                    exit()
                if done:
                    print("episode: {}, steps: {}, total_reward: {}".format(i, n_steps, total_reward))
                    total_reward_list.append(total_reward)
                    steps_list.append(n_steps)
                    break
                agent.update(pre_obs, action, ob, reward, done, option)
                if agent.get_terminate(ob, option):
                    option = agent.get_option(ob)
                
                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        # Close the env and write monitor result info to disk
    except KeyboardInterrupt:
        pass
    date = datetime.now().strftime("%Y%m%d")
    time = datetime.now().strftime("%H%M")
    saved_dir = os.path.join("data", date, time)
    # export process
    export_csv(saved_dir, "total_reward.csv", total_reward_list)
    td_error_list = agent.td_error_list
    export_csv(saved_dir, "td_error.csv", td_error_list)
    total_reward_list = np.array(total_reward_list)
    steps_list = np.array(steps_list)
    x = list(range(len(total_reward_list)))
    plt.subplot(2,2,1)
    y = moved_average(total_reward_list, 10)
    plt.plot(x, total_reward_list)
    plt.plot(x, y, 'r--')
    plt.subplot(2,2,2)
    y = moved_average(steps_list, 10)
    plt.plot(x, steps_list)
    plt.plot(x, y, 'r--')

    plt.subplot(2,1,2)
    y = moved_average(td_error_list, 1000)
    x = list(range(len(td_error_list)))
    plt.plot(x, td_error_list, 'k-')
    plt.plot(x, y, 'r--', label='average')
    plt.legend()
    plt.show()
    env.close()
