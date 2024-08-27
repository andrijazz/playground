import gymnasium as gym
import torch
import random


# The network is trained to predict the expected value for each action, given the input state. The action with the
# highest expected value is then chosen.
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# DQN algorithm
# 1. init replay buffer, policy_net, target_net
# 2. populate replay buffer with transitions (s, a, r, s')
# 3. training agent
#       - sample random batch of transitions
#       - Q*(s, a) -> E(R), E(R) = r + gamma * Q(s', a')
#       - policy_net(s, a) -> E(R)
#       - target_net (s') -> Q(s', a') * gamma + r
#       - update rule

class ReplayBuffer:
    def __init__(self, N):
        self.N = N
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.N:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.N

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def test_replay_buffer():
    replay_buffer = ReplayBuffer(100)
    env = gym.make('CartPole-v1', render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        replay_buffer.add(observation, action, reward, next_observation, terminated)
        if terminated or truncated:
            observation, info = env.reset()
        else:
            observation = next_observation
    env.close()

    assert len(replay_buffer.buffer) == 100

test_replay_buffer()

# observation, info = env.reset(seed=42)
# i = 0
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()

