# _*_ coding utf-8 _*_
"""
@File : DDPG.py
@Author: yxwang
@Date : 2020/4/9
@Desc :
"""

import tensorflow as tf
import huskarl as hk
import gym
import matplotlib.pyplot as plt

# from BlockChain import BlockChainEnv
from keras.layers import Lambda

if __name__ == "__main__":

    # Setup Block_chain environment
    create_env = lambda: gym.make('BlockChain-v0')
    dummy_env = create_env()

    action_size = dummy_env.n_of_nodes + 3

    shape0 = 0
    for k, v in dummy_env.observation_space.spaces.items():
        shape = v.shape
        shape0 += shape[0]
    state_shape = (shape0,)

    # create_env = lambda: gym.make('Pendulum-v0')
    # dummy_env = create_env()
    # action_size = dummy_env.action_space.shape[0]
    # state_shape = dummy_env.observation_space.shape

# Build a simple actor model which is function estimator of policy
# actor = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(16, activation='relu', input_shape=state_shape),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(action_size, activation='relu')
# ])
# bound = 1

inputs = tf.keras.Input(shape=state_shape, name='state_input')
x = tf.keras.layers.Dense(16, activation='relu')(inputs)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
# output = Lambda(lambda x: x*bound)
output = tf.keras.layers.Dense(action_size, activation='linear')(x)
actor = tf.keras.Model(inputs=inputs, outputs=output)

# Build a simple critic model which is the function estimator of value
action_input = tf.keras.Input(shape=(action_size,), name='action_input')
state_input = tf.keras.Input(shape=state_shape, name='state_input')
x = tf.keras.layers.Concatenate()([action_input, state_input])
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(1, activation='linear')(x)
critic = tf.keras.Model(inputs=[action_input, state_input], outputs=x)

# Create Deep Deterministic Policy Gradient agent
agent = hk.agent.DDPG(actor=actor, critic=critic, nsteps=2)


# def plot_rewards(episode_rewards, episode_steps, done=False):
#     plt.clf()
#     plt.xlabel('Step')
#     plt.ylabel('Reward')
#     for ed, steps in zip(episode_rewards, episode_steps):
#         plt.plot(steps, ed)
#     plt.show() if done else plt.pause(0.001)  # Pause a bit so that the graph is updated


# Create simulation, train and then test
sim = hk.Simulation(create_env, agent)
sim.train(max_steps=30, visualize=False, plot=None)
# sim.train(max_steps=30_000, visualize=False, plot=plot_rewards)
# sim.test(max_steps=5_000)
