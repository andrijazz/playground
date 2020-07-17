from __future__ import absolute_import, division

import copy

import torch
import torch.nn as nn
import torch.optim as optim

import core.factory as factory
import core.utils as utils
import projects.mlp.pth_utils as pth_utils
from core.base_policy import BasePolicy


class MLPPolicy(BasePolicy):
    def __init__(self):
        super(MLPPolicy).__init__()
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.build_model()

    def build_model(self):
        # construct the net
        mlp_config = utils.get_config('projects.mlp.mlp_config')
        self.net = factory.create_net(mlp_config)
        params_to_update = self.net.parameters()
        self.optimizer = optim.Adam(params_to_update, lr=mlp_config.TRAIN_LR)
        self.criterion = nn.MSELoss()

    def restore(self, filepath, storage):
        checkpoint = pth_utils.restore_model(filepath, storage)
        self.net.load_state_dict(checkpoint['state_dict'])

    def save(self, filename, step, upload_to_wandb=False):
        checkpoint = {'step': step, 'state_dict': copy.deepcopy(self.net.state_dict())}
        return pth_utils.save_model(checkpoint, filename, upload_to_wandb)

    def get_action(self, inputs):
        inputs = torch.from_numpy(inputs)
        inputs = inputs.reshape(1, inputs.shape[0]).float()
        # forward
        outputs = self.net(inputs)
        return outputs.detach().numpy().reshape(-1)

    def update(self, inputs, targets):
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        self.optimizer.zero_grad()
        # forward
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
