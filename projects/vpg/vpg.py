import argparse
import os
import random
import gym
import torch
import torch.nn as nn
import wandb
import yaml
import numpy as np
from torch.optim import SGD

# wandb
# enable dryrun to turn off wandb syncing completely
# os.environ['WANDB_MODE'] = 'dryrun'
# prevent wandb uploading pth to cloud
os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'


class ReplayBuffer:
    """
    Replay buffer of trajectories
    """
    def __init__(self, max_size=10000):
        # list of trajectories s, a, s', r, t
        self.buffer = list()
        for i in range(max_size):
            self.buffer.append([])
        self.max_size = max_size
        self.curr = 0

    def add(self, trajectory):
        if self.curr >= self.max_size:
             self.curr = 0
        self.buffer[self.curr] = trajectory
        self.curr += 1

    def sample(self):
        index = random.random(self.curr)
        return self.buffer[index]


class MLP(nn.Module):
    def __init__(self, n_in, n_out, arch):
        super(MLP, self).__init__()
        assert arch is not None
        assert len(arch) > 0

        self.net = nn.Sequential()
        prev = n_in
        for i, m in enumerate(arch):
            self.net.add_module(f'dense{i}', nn.Linear(prev, int(m)))
            self.net.add_module(f'dense{i}_act', nn.ReLU())
            prev = int(m)
        self.net.add_module(f'dense_{len(arch)}', nn.Linear(prev, n_out))

    def forward(self, x):
        return self.net(x)


class VPGPolicy:
    def __init__(self, n_in, n_out, config):
        self.config = config
        self.net = MLP(n_in, n_out, self.config['arch'])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.net.parameters(), lr=self.config['lr'])

    def get_action(self, observation):
        # convert to tensor
        observation = torch.as_tensor(observation, dtype=torch.float)
        output = self.net(observation)
        dist = torch.distributions.Categorical(logits=output)
        # sample from policy
        action = dist.sample()
        # greedy action
        # action = torch.argmax(torch.softmax(output, dim=0))

        # convert to numpy array
        return action.item()

    def update(self, observations, actions, rewards):
        # convert to tensors
        observations = torch.as_tensor(observations, dtype=torch.float)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float)

        logits = self.net(observations)
        # creates distribution by normalizing logits to sum to 1
        dist = torch.distributions.Categorical(logits=logits)
        J = torch.sum(rewards)
        loss = -(dist.log_prob(actions) * torch.sum(rewards)).sum()

        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return J.item(), loss.item()

    def save(self, save_path):
        pass

    def restore(self, restore_path):
        pass


def unpack_trajectories(trajectories):
    s = list()
    a = list()
    s_p = list()
    r = list()
    d = list()
    for trajectory in trajectories:
        for t in trajectory:
            s.append(t[0])
            a.append(t[1])
            s_p.append(t[2])
            r.append(t[3])
            d.append(int(t[4]))
    s = np.stack(s, axis=0)
    a = np.stack(a, axis=0)
    s_p = np.stack(s_p, axis=0)
    r = np.stack(r, axis=0)
    d = np.stack(d, axis=0)

    assert s.shape[0] == a.shape[0] # ...
    return s, a, s_p, r, d


def main(args):
    # load config
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        config = config['config']

    # init wandb
    wandb.init(project='vpg', config=config, dir=os.getenv('LOG'))

    # create env
    env = gym.make(config['env'])
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n

    # create policy
    pi = VPGPolicy(n_in, n_out, config)

    # training loop
    for i_episode in range(config['num_episodes']):
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
                break

        print("Episode finished after {} timesteps".format(t + 1))
        print("Performing grad update")
        observations, actions, observations_n, rewards, dones = unpack_trajectories([trajectory])
        J, loss = pi.update(observations, actions, rewards)
        wandb.log({
            'J': J,
            'loss': loss,
            't': t
        })
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file', default='./config/dev_config.yml')
    args = parser.parse_args()
    main(args)
