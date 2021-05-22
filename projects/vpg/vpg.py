import argparse
import os
import random
import gym
import torch
import torch.nn as nn
import wandb
import yaml
import numpy as np
from torch.optim import Adam
from torch.distributions import Categorical

# wandb
# enable dryrun to turn off wandb syncing completely
# os.environ['WANDB_MODE'] = 'dryrun'
# prevent wandb uploading pth to cloud
os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'

from itertools import accumulate
import functools
from typing import List


# TODO
# 1. baselines
# 2. standardize advantage???
# 3. calc return calc cumsum
# 4. continuous action space
# 5. write readme + doc what i have learned and experiments
# 6. gae-lambda
# 7. parallel rollouts


def _discounted_return(rewards: List[float], gamma: float) -> List[float]:
    """
    Helper function
    Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single
    rollout of length T
    Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
    """

    discounted_return = functools.reduce(
        lambda ret, reward: ret * gamma + reward,
        reversed(rewards),
    )
    return [discounted_return] * len(rewards)


def _discounted_cumsum(rewards: List[float], gamma: float) -> List[float]:
    """
        Helper function which
        - takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        - and returns a list where the entry in each index t' is
          sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    """

    return list(accumulate(
        reversed(rewards),
        lambda ret, reward: ret * gamma + reward,
    ))[::-1]


def load_config(yml_config_file):
    try:
        with open(yml_config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            return config['config']
    except FileNotFoundError:
        print('Config file {} not found'.format(yml_config_file))
        exit(1)


class ReplayBuffer:
    """
    Replay buffer of trajectories
    """
    def __init__(self, max_size=10000):
        # list of trajectories s, a, s', r, t
        self.trajectories = dict()
        self.max_size = max_size
        self.curr = 0
        self.collected_steps = 0
        self.is_full = False

    def add(self, trajectory):
        if self.curr >= self.max_size:
            self.curr = 0
            self.is_full = True

        # if replacing old trajectory refresh number of collected steps
        old_trajectory = self.trajectories.get(self.curr, None)
        if old_trajectory:
            self.collected_steps -= len(old_trajectory)

        self.trajectories[self.curr] = trajectory
        self.collected_steps += len(trajectory)
        self.curr += 1

    def sample_recent(self, batch_size):
        if len(self.trajectories) == 0:
            return []

        i = self.curr - 1 if self.curr > 0 else len(self.trajectories) - 1
        k = 0
        indices = []
        while k < batch_size:
            k += len(self.trajectories[i])
            indices.append(i)
            i = i - 1 if i > 0 else len(self.trajectories) - 1

        sampled_trajectories = [self.trajectories.get(i) for i in indices]
        return sampled_trajectories

    def sample_random(self, batch_size):
        if len(self.trajectories) == 0:
            return []

        k = 0
        indices = []
        while k < batch_size:
            i = random.sample(range(len(self.trajectories)))
            k += len(self.trajectories[i])
            indices.append(i)
        sampled_trajectories = [self.trajectories.get(i) for i in indices]
        return sampled_trajectories

    # def sample_random_trajectories(self, n):
    #     if n > len(self.trajectories):
    #         n = len(self.trajectories)
    #     indices = random.sample(range(len(self.trajectories)), n)
    #     sampled_trajectories = [self.trajectories.get(i) for i in indices]
    #     return sampled_trajectories
    #
    # def sample_recent_trajectories(self, n):
    #     if n > len(self.trajectories):
    #         n = len(self.trajectories)
    #     d = self.curr - n
    #     if d >= 0:
    #         indices = list(range(self.curr - n, self.curr))
    #     else:
    #         indices = list(range(len(self.trajectories) + d, len(self.trajectories)))
    #         indices.extend(list(range(self.curr)))
    #
    #     sampled_trajectories = [self.trajectories.get(i) for i in indices]
    #     return sampled_trajectories

    def concatenate_trajectories(self, trajectories):
        s = list()
        a = list()
        s_p = list()
        r = list()
        d = list()
        u_r = list()    # unconcatenated reward
        for trajectory in trajectories:
            u_r_t = list()
            for t in trajectory:
                s.append(t[0])
                a.append(t[1])
                s_p.append(t[2])
                r.append(t[3])
                d.append(int(t[4]))
                u_r_t.append(t[3])
            u_r.append(np.array(u_r_t))
        s = np.stack(s, axis=0)
        a = np.stack(a, axis=0)
        s_p = np.stack(s_p, axis=0)
        r = np.stack(r, axis=0)
        d = np.stack(d, axis=0)

        assert s.shape[0] == a.shape[0] # ...
        return s, a, s_p, r, d, u_r

    def num_of_trajectories(self):
        return len(self.trajectories)

    def num_of_collected_steps(self):
        return self.collected_steps


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class MLP(nn.Module):
    def __init__(self, n_in, n_out, arch, activation):
        super(MLP, self).__init__()
        assert arch is not None
        assert len(arch) > 0

        self.net = nn.Sequential()
        prev = n_in
        for i, m in enumerate(arch):
            self.net.add_module(f'dense{i}', nn.Linear(prev, int(m)))
            self.net.add_module(f'dense{i}_act', _str_to_activation[activation])
            prev = int(m)
        self.net.add_module(f'dense_{len(arch)}', nn.Linear(prev, n_out))

    def forward(self, x):
        return self.net(x)


class VPGPolicy:
    def __init__(self, n_in, n_out, config):
        self.config = config
        self.net = MLP(n_in, n_out, self.config['arch'], self.config['activation'])
        self.optimizer = Adam(self.net.parameters(), lr=self.config['lr'])

    def get_action(self, observation):
        # collecting experience ... check if net is in eval mode
        if self.net.training:
            self.net.eval()

        # convert to tensor
        observation = torch.as_tensor(observation, dtype=torch.float)
        logit = self.net(observation)
        m = Categorical(logits=logit)
        action = m.sample()
        # greedy action
        # action = torch.argmax(torch.softmax(output, dim=0))
        # convert to numpy array
        return action.item()

    def update(self, observations, actions, u_rewards):
        if not self.net.training:
            self.net.train()

        # calculate q's
        if self.config['reward_to_go']:
            q_values = np.concatenate([_discounted_cumsum(r, gamma=self.config['gamma']) for r in u_rewards])
        else:
            q_values = np.concatenate([_discounted_return(r, gamma=self.config['gamma']) for r in u_rewards])

        if self.config['baselines']:
            advantages = None

        g = [np.sum(r) for r in u_rewards]
        g_avg = np.mean(g)
        g_max = np.max(g)

        # convert to tensors
        observations = torch.as_tensor(observations, dtype=torch.float)
        actions = torch.as_tensor(actions, dtype=torch.int8)
        q_values = torch.as_tensor(q_values, dtype=torch.float)

        logits = self.net(observations)
        m = Categorical(logits=logits)
        loss = -torch.sum(m.log_prob(actions) * q_values)

        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return g_avg, g_max, loss.item()

    def save(self, save_path, step):
        with torch.no_grad():
            model_name = '{}-{}.pth'.format(self.config['model'], step)
            checkpoint_file = os.path.join(save_path, model_name)
            torch.save(self.net.state_dict(), checkpoint_file)

    def restore(self, restore_path):
        self.net.load_state_dict(torch.load(restore_path))


def train(args):
    config = load_config(args.config)

    # init wandb
    wandb.init(project='vpg', config=config, dir=os.getenv('LOG'))

    # create env
    env = gym.make(config['env'])
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n

    # create policy
    pi = VPGPolicy(n_in, n_out, config)

    # replay buffer
    replay_buffer = ReplayBuffer()

    # training loop
    for i in range(config['num_iterations']):
        # Step 1. Data Collection
        n = 0   # number of collected steps per iter
        while n < config['batch_size']:
            # collect single trajectory
            observation = env.reset()
            t = 0
            trajectory = []
            while True:
                env.render()
                # sampling random action
                # action = env.action_space.sample()
                action = pi.get_action(observation)
                observation_n, reward, done, info = env.step(action)
                trajectory.append((observation, action, observation_n, reward, done))
                observation = observation_n
                t += 1
                if done:
                    print("Episode finished after {} timesteps".format(t))
                    replay_buffer.add(trajectory)
                    n += len(trajectory)
                    break

        # Step 2. Policy update
        trajectories = replay_buffer.sample_recent(config['batch_size'])
        observations, actions, observations_n, rewards, dones, u_rewards = replay_buffer.concatenate_trajectories(trajectories)
        g_avg, g_max, loss = pi.update(observations, actions, u_rewards)
        wandb.log({
            'g_avg': g_avg,
            'g_max': g_max,
            'loss': loss
        })

        # save policy
        pi.save(wandb.run.dir, i)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file', default='./config/dev_config.yml')
    args = parser.parse_args()
    train(args)
