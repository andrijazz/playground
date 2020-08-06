from __future__ import absolute_import, division

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import core.factory as factory
import projects.mlp.pth_utils as pth_utils
from core.base_model import BaseModel
from core.utils import accuracy, is_debug_session
from projects.mlp.mlp_datasets import create_train_and_val_datasets, create_test_dataset


class MLPModel(BaseModel):
    """
    Model for mlp net
    """
    def __init__(self, config):
        super().__init__(config)

        # construct the model
        self.net = factory.create_net(self.config)

    def restore(self, filepath, storage):
        checkpoint = pth_utils.restore_model(filepath, storage)
        self.net.load_state_dict(checkpoint['state_dict'])

    def save(self, filename, step, upload_to_wandb=False):
        checkpoint = {'step': step, 'state_dict': copy.deepcopy(self.net.state_dict())}
        return pth_utils.save_model(checkpoint, filename, upload_to_wandb)

    def train(self):
        wandb.init(project=os.getenv('PROJECT'), dir=os.getenv('LOG'), config=self.config, reinit=True)

        if self.config.TRAIN_RESTORE_FILE:
            self.restore(self.config.TRAIN_RESTORE_FILE, self.config.TRAIN_RESTORE_STORAGE)

        kwargs = {'num_workers': 8, 'pin_memory': True} \
            if torch.cuda.is_available() and not is_debug_session() else {}

        train_dataset, val_dataset = create_train_and_val_datasets(self.config.TRAIN_DATASET)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.config.TRAIN_BATCH_SIZE,
                                                   shuffle=True,
                                                   **kwargs)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.config.VAL_BATCH_SIZE,
                                                 shuffle=False,
                                                 **kwargs)

        device = self.config.GPU
        self.net = self.net.to(device)
        params_to_update = self.net.parameters()
        optimizer = optim.Adam(params_to_update, lr=self.config.TRAIN_LR)
        criterion = nn.CrossEntropyLoss()
        step = 0

        best_checkpoint_info = {'step': step, 'loss': np.inf, 'acc': 0}

        # set model to train mode
        self.net.train()

        for epoch in range(self.config.TRAIN_NUM_EPOCHS):

            wandb.log({"train/epoch": epoch}, step=step)

            for samples in train_loader:
                inputs = samples[0]
                batch_size = inputs.shape[0]
                in_dim = inputs.shape[2] * inputs.shape[3]
                inputs_vec = inputs.reshape((batch_size, in_dim))
                labels = samples[1]
                inputs_vec = inputs_vec.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = self.net(inputs_vec)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                probability_outputs = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probability_outputs, dim=1)
                if step % self.config.TRAIN_SUMMARY_FREQ == 0:
                    # log scalars
                    wandb.log({"train/loss": loss}, step=step)

                    # plot random sample and predicted class
                    sample_idx = np.random.choice(batch_size)
                    predicted_caption = str(predicted_classes[sample_idx].item())
                    gt_caption = str(labels[sample_idx].item())
                    caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)
                    wandb.log({"train/samples": wandb.Image(inputs[sample_idx], caption=caption)}, step=step)

                if step % self.config.TRAIN_VAL_FREQ == 0:
                    self.net.eval()
                    val_loss, val_acc = self._validate(criterion, val_loader, step, device)
                    wandb.log({"val/loss": val_loss}, step=step)
                    wandb.log({"val/accuracy": val_acc}, step=step)

                    if val_acc > best_checkpoint_info['acc'] and epoch > self.config.TRAIN_START_SAVING_AFTER_EPOCH:
                        best_checkpoint_info['loss'] = val_loss
                        best_checkpoint_info['acc'] = val_acc
                        best_checkpoint_info['step'] = step
                        checkpoint_file = self.save('model', step)
                        best_checkpoint_info['checkpoint_file'] = checkpoint_file

                    self.net.train()

                if step % self.config.TRAIN_SAVE_MODEL_FREQ == 0:
                    self.save('checkpoint', step)

                step += 1

        # restore best checkpoint state
        self.restore(best_checkpoint_info['checkpoint_file'], storage='local')
        # save best model as {model_name}.pth and upload it to wandb if specified
        model_name = self.config.MODEL.lower()
        self.save(model_name, best_checkpoint_info['step'], upload_to_wandb=self.config.UPLOAD_BEST_TO_WANDB)
        return best_checkpoint_info['acc']

    def test(self):
        wandb.init(project=os.getenv('PROJECT'), dir=os.getenv('LOG'), config=self.config, reinit=True)

        if self.config.TEST_RESTORE_FILE:
            self.restore(self.config.TEST_RESTORE_FILE, self.config.TEST_RESTORE_STORAGE)

        kwargs = {'num_workers': 8, 'pin_memory': True} \
            if torch.cuda.is_available() and not is_debug_session() else {}

        dataset = create_test_dataset(self.config.TEST_DATASET)

        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.config.TEST_BATCH_SIZE,
                                                  shuffle=False,
                                                  **kwargs)
        criterion = nn.CrossEntropyLoss()
        device = self.config.GPU
        self.net = self.net.to(device)
        num_of_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('Total number of parameters is {}'.format(num_of_params))
        self.net.eval()
        loss, acc = self._validate(criterion, test_loader, 0, device)
        wandb.log({'test/loss': loss, 'test/acc': acc})

    def _validate(self, criterion, val_loader, step, device):
        loss_meter = pth_utils.AverageMeter()
        p = []
        gt = []
        for samples in val_loader:
            inputs = samples[0]
            batch_size = inputs.shape[0]
            in_dim = inputs.shape[2] * inputs.shape[3]
            inputs_vec = inputs.reshape((batch_size, in_dim))
            labels = samples[1]
            inputs_vec = inputs_vec.to(device)
            labels = labels.to(device)

            # forward
            outputs = self.net(inputs_vec)
            loss = criterion(outputs, labels)

            probability_outputs = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probability_outputs, dim=1)
            p.extend(predicted_classes.tolist())
            gt.extend(labels.tolist())
            loss_meter.update(loss.item(), batch_size)

        p = np.array(p, dtype=np.int)
        gt = np.array(gt, dtype=np.int)
        acc = accuracy(p, gt)
        val_loss = loss_meter.avg
        return val_loss, acc

    def inference(self):
        pass
