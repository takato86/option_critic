import gym
import gym_pinball

env = gym.make('PinBall-v0')
env.reset()
while 1:
    env.render()