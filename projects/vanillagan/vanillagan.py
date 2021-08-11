import argparse
import os
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from core.utils.utils import get_config_yml, is_debug_session


class Maxout(nn.Module):
    """
    Maxout non-linearity

    References:
        [original paper] https://arxiv.org/abs/1302.4389
        [random impl] https://github.com/pytorch/pytorch/issues/805#issuecomment-389447728
        [random impl] https://github.com/paniabhisek/maxout/blob/master/maxout.py
    """
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m


class Generator(nn.Module):
    """
    Generator network as described in the official implementation [1].

    Additionally, layers from original implementation are implemented here [2].

    References:
        [1] https://github.com/goodfeli/adversarial/blob/master/cifar10_fully_connected.yaml
        [2] https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/models/mlp.py#L2493
    """

    def __init__(self, z_dim=8000, hidden_dim=8000, output_dim=3072):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        layers = [
            ('gh0', nn.Linear(self.z_dim, self.hidden_dim)),
            ('gh0_act', nn.ReLU()),
            ('h1', nn.Linear(self.hidden_dim, self.hidden_dim)),
            ('h1_act', nn.Sigmoid()),
            ('y', nn.Linear(self.hidden_dim, self.output_dim)),
            ('y_act', nn.Tanh())
        ]
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
    Discriminator network as described in the official implementation [1].

    Additionally, layers from original implementation are implemented here [2].

    References:
        [1] https://github.com/goodfeli/adversarial/blob/master/cifar10_fully_connected.yaml
        [2] https://github.com/lisa-lab/pylearn2/blob/58ba37286182817301ed72b0f143a89547b3f011/pylearn2/models/maxout.py#L60
    """

    def __init__(self, input_dim=3072, hidden_dim=1600, output_dim=1, pool_size=5):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pool_size = pool_size

        h1_dim = self.hidden_dim // self.pool_size
        y_dim = h1_dim // self.pool_size
        layers = [
            ('dh0', nn.Linear(self.input_dim, self.hidden_dim)),
            ('dh0_act', Maxout(pool_size=self.pool_size)),
            ('h1', nn.Linear(h1_dim, h1_dim)),
            ('h1_act', Maxout(pool_size=self.pool_size)),
            ('y', nn.Linear(y_dim, self.output_dim)),
            ('y_act', nn.Sigmoid())
        ]

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)


def build_dataset(config):
    if config['dataset'] == "cifar10":
        # setup transforms
        transform = transforms.Compose([
            transforms.Resize(config.get("resolution")),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(os.getenv("DATASETS"), transform=transform, download=True)
        return train_dataset
    else:
        exit(f"Unsupported dataset {config['dataset']} specified")


def train(config):
    # init wandb
    wandb.init(project='vanillagan', config=config, dir=os.getenv('LOG'))

    # init models
    generator = Generator()
    generator.to(config['device'])
    discriminator = Discriminator()
    discriminator.to(config['device'])

    # speed up settings
    cudnn.benchmark = True

    # setup dataset
    train_dataset = build_dataset(config)

    kwargs = {}
    if torch.cuda.is_available() and not is_debug_session():
        kwargs = {'num_workers': 2, 'pin_memory': True}

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.get("train_batch_size"),
                                  drop_last=True,
                                  shuffle=True,
                                  **kwargs)

    adversarial_criterion = nn.BCELoss()
    d_optim = torch.optim.SGD(discriminator.parameters(), lr=config['discriminator_lr'])
    g_optim = torch.optim.SGD(generator.parameters(), lr=config['generator_lr'])

    reals_gt = torch.ones((config['train_batch_size'], 1), device=config['device'])
    fakes_gt = torch.zeros((config['train_batch_size'], 1), device=config['device'])

    # setup train mode
    generator.train()
    discriminator.train()

    batch_size = config['train_batch_size']
    step = 0
    for epoch in range(config['num_epochs']):
        for samples in train_dataloader:
            reals = samples[0]
            reals = reals.to(config['device'])

            z = torch.rand((batch_size, generator.z_dim), device=config['device'])
            fakes = generator(z)

            # discriminator loss
            # reals log(D(x))
            d_reals = discriminator(reals.reshape((batch_size, 3072)))
            reals_score = adversarial_criterion(input=d_reals, target=reals_gt)
            # Pay attention to .detach(). Thats because we want to update only discriminator parameters
            # fakes log(1-D(G(z)))
            d_fakes = discriminator(fakes.detach())
            fakes_score = adversarial_criterion(input=d_fakes, target=fakes_gt)
            d_loss = reals_score + fakes_score

            # update discriminator
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            log_dict = {
                'reals_score': reals_score.item(),
                'fakes_score': fakes_score.item(),
                'd_loss': d_loss.item(),
            }

            # every k-th step update generator
            if step % config['k'] == 0:
                # generator loss
                d_fakes = discriminator(fakes)
                g_loss = adversarial_criterion(input=d_fakes, target=reals_gt)

                # update generator
                g_optim.zero_grad()
                g_loss.backward()
                g_optim.step()

                log_dict['g_loss'] = g_loss.item()

            if step % config['train_log_freq'] == 0:
                # log fakes
                images = fakes.view(batch_size, 3, config['resolution'], config['resolution'])
                images.clamp_(-1, 1)
                images.add_(1).div_(2)
                grid_image = torchvision.utils.make_grid(images.detach().cpu(), nrow=10)
                log_dict["fakes"] = wandb.Image(grid_image)

                # TODO calculate fid / inception score
                wandb.log(log_dict)

            step += 1


def main(args):
    # load config
    config = get_config_yml(args.config)

    if args.generate:
        # generate samples
        pass
    else:
        train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vanilla GAN on cifar10")
    parser.add_argument('--config', type=str, help='Path to config file', default='./config/dev_config.yml')
    parser.add_argument("--generate", action="store_true", default=False, help="Run in inference mode")
    args = parser.parse_args()
    main(args)
