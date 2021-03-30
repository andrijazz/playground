import argparse
import os

import gym
import torch
import torch.nn as nn
import wandb
import yaml
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
        self.buffer = list()
        self.max_size = max_size
        self.curr = 0
        # s, a, s', r, t
        pass

    def add(self, trajectory):
        if len(self.buffer) > self.max_size:
            pass


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
        output = self.net(observation)
        action = torch.argmax(torch.softmax(output, dim=0))
        return action

    def update(self, observations, target_actions, rewards):
        outputs = self.net(observations)
        loss = self.criterion(outputs, target_actions)  # + rewards

        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        wandb.log({'loss': loss})

    def save(self, save_path):
        pass

    def restore(self, restore_path):
        pass


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

        trajectory = ReplayBuffer()
        t = 0
        while True:
            env.render()
            # sampling random action
            # action = env.action_space.sample()
            action = pi.get_action(torch.as_tensor(observation, dtype=torch.float))
            observation, reward, done, info = env.step(action.detach().numpy())
            t += 1
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print("Performing grad update")
                # pi.update()
                break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file', default='./config/dev_config.yml')
    args = parser.parse_args()
    main(args)
