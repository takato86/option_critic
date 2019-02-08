import gym
import numpy as np
import tensorflow as tf
from gym import wrappers
from replay_buffer import ReplayBuffer
from option_critic import OptionCritic
from state_processor import StateProcessor
from tensorflow.python import debug as tf_debug

MAX_EPISODES = int(8e3)
MAX_STEPS_PER_EPISODE = 25e4
PRE_TRAIN_STEPS = 32

def get_epsilon(frame_count):
    # TODO
    return 0.9

def get_reward(score):
    # TODO
    """
    reward 
    """
    if score > 0:
        reward = 1
    elif score < 0:
        reward = -1
    else:
        reward = 0
    return reward

def train(sess, env, option_critic, replay_buffer):

    state_processor = StateProcessor(sess)
    frame_count = 0
    update_freq = 32
    for episode_counter in range(MAX_EPISODES):
        start_frame = frame_count
        current_option = 0
        current_state = env.reset()
        # TODO state_processorの実装
        current_state = state_processor.process(current_state)
        # TODO np.stackの使い方 4 frame skipping: 4フレームは同じ行動を取り続ける
        current_state = np.stack([current_state]*4, axis=0)
        done = False
        termination = False
        print("episode: {}".format(episode_counter))
        while MAX_STEPS_PER_EPISODE > (frame_count - start_frame) or not done:            
            frame_count += 1
            print("step: {}".format(frame_count))
            eps = get_epsilon(frame_count)
            action_probs = option_critic.take_intra_option(current_state)
            current_action = np.argmax(np.random.multinomial(1, action_probs[0]))

            next_state, score, done, info = env.step(current_action)
            next_state = state_processor.process(next_state)
            next_state = np.append(current_state[1:,:,:],
                                    np.expand_dims(next_state, 0),
                                    axis=0)
            reward = get_reward(score)
            replay_buffer.add_experience(current_state, current_option, current_state, reward, next_state, termination)

            term_prob = option_critic.take_termination(current_state)[current_option]
            randomize = np.random.rand()
            termination = True if term_prob > randomize else False

            if frame_count < PRE_TRAIN_STEPS:
                termination = True
            else:
                option_critic.update_actor(current_state, current_option, current_action, next_state, reward)

            if frame_count % update_freq == 0:
                option_critic.update_critic()

            if termination:
                current_option = option_critic.take_option(current_state)
            
            current_state = next_state
        


def main(_):
    state_dim =  env.observation_space.shape[0]
    action_dim = env.action_space.n
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    replay_buffer = ReplayBuffer()
    option_critic = OptionCritic(sess, 8, 18, replay_buffer, 4, 32, 0.9, 0.01, epsilon=0.9)
    init = tf.global_variables_initializer()
    sess.run(init)
    train(sess, env, option_critic, replay_buffer)

if __name__=="__main__":
    env = gym.make("Seaquest-v0")
    env.reset()
    tf.app.run()