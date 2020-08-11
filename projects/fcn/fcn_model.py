from __future__ import absolute_import, division

import os

import numpy as np
import tensorflow as tf
import wandb

import core.factory as factory
from core.base_model import BaseModel
from core.logger import get_logger
from projects.fcn.fcn_datasets import create_train_and_val_datasets

logger = get_logger()


class FCNModel(BaseModel):
    """
    Model for fcn net
    """
    def __init__(self, config):
        super().__init__(config)

        # construct the model
        self.net = factory.create_net(self.config)

    def restore(self, filename, storage):
        if storage == 'wandb':
            parts = filename.split('/')
            wandb_path = '/'.join(parts[:-1])
            wandb_file = parts[-1]
            restore_file = wandb.restore(wandb_file, run_path=wandb_path)
            self.net.load_weights(restore_file)
        elif storage == 'local':  # local storage
            self.net.load_weights(filename)
        else:
            print('Unknown storage type')

    def save(self, filename, step, upload_to_wandb=False):
        model_name = '{}-{}'.format(filename, step)
        checkpoint_file = os.path.join(wandb.run.dir, model_name)
        if upload_to_wandb:
            wandb.save(model_name)
        self.net.save_weights(checkpoint_file)

    def train(self):
        wandb.init(project=os.getenv('PROJECT'), dir=os.getenv('LOG'), config=self.config, reinit=True)

        if self.config.TRAIN_RESTORE_FILE:
            self.restore(self.config.TRAIN_RESTORE_FILE, self.config.TRAIN_RESTORE_STORAGE)

        train_dataset = create_train_and_val_datasets(self.config.TRAIN_DATASET)
        train_dataset = train_dataset.shuffle(len(list(train_dataset)))
        train_dataset = train_dataset.batch(self.config.TRAIN_BATCH_SIZE)
        logger.info('Training dataset size = {}'.format(len(list(train_dataset))))

        # val_dataset = val_dataset.shuffle(len(images))
        # val_dataset = val_dataset.batch(self.config.TRAIN_BATCH_SIZE)
        # logger.info('Validation dataset size = {}'.format(len(list(train_dataset))))

        # Instantiate an optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.TRAIN_LR)
        # Instantiate a loss function.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        step = 0

        best_checkpoint_info = {'step': step, 'loss': np.inf, 'acc': 0}

        for epoch in range(self.config.TRAIN_NUM_EPOCHS):

            wandb.log({"train/epoch": epoch}, step=step)

            for (x_batch_train, y_batch_train, y_batch_train_idx) in train_dataset:
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.

                    # Logits for this minibatch
                    logits = self.net(x_batch_train, training=True)

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train_idx, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.net.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, self.net.trainable_weights))

                if step % self.config.TRAIN_SUMMARY_FREQ == 0:
                    # log scalars
                    wandb.log({"train/loss": loss_value}, step=step)

                    # log images
                    wandb.log({"train/image": wandb.Image(x_batch_train[0])}, step=step)
                    wandb.log({"train/gt_image": wandb.Image(y_batch_train[0])}, step=step)
                    # wandb.log({"train/y_pred": wandb.Image(y_batch_train[0])}, step=step)

                if step % self.config.TRAIN_VAL_FREQ == 0:
                    pass
                    # self.net.eval()
                    # val_loss, val_acc = self._validate(criterion, val_loader, step, device)
                    # wandb.log({"val/loss": val_loss}, step=step)
                    # wandb.log({"val/accuracy": val_acc}, step=step)
                    #
                    # if val_acc > best_checkpoint_info['acc'] and epoch > self.config.TRAIN_START_SAVING_AFTER_EPOCH:
                    #     best_checkpoint_info['loss'] = val_loss
                    #     best_checkpoint_info['acc'] = val_acc
                    #     best_checkpoint_info['step'] = step
                    #     checkpoint_file = self.save('model', step)
                    #     best_checkpoint_info['checkpoint_file'] = checkpoint_file
                    #
                    # self.net.train()

                if step % self.config.TRAIN_SAVE_MODEL_FREQ == 0:
                    pass
                    # self.save('checkpoint', step)

                step += 1

        # # restore best checkpoint state
        # self.restore(best_checkpoint_info['checkpoint_file'], storage='local')
        # # save best model as {model_name}.pth and upload it to wandb if specified
        # model_name = self.config.MODEL.lower()
        # self.save(model_name, best_checkpoint_info['step'], upload_to_wandb=self.config.UPLOAD_BEST_TO_WANDB)
        # return best_checkpoint_info['acc']

    def test(self):
        pass
        # wandb.init(project=os.getenv('PROJECT'), dir=os.getenv('LOG'), config=self.config, reinit=True)
        #
        # if self.config.TEST_RESTORE_FILE:
        #     self.restore(self.config.TEST_RESTORE_FILE, self.config.TEST_RESTORE_STORAGE)
        #
        # kwargs = {'num_workers': 8, 'pin_memory': True} \
        #     if torch.cuda.is_available() and not is_debug_session() else {}
        #
        # dataset = create_test_dataset(self.config.TEST_DATASET)
        #
        # test_loader = torch.utils.data.DataLoader(dataset,
        #                                           batch_size=self.config.TEST_BATCH_SIZE,
        #                                           shuffle=False,
        #                                           **kwargs)
        # criterion = nn.CrossEntropyLoss()
        # device = self.config.GPU
        # self.net = self.net.to(device)
        # num_of_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        # print('Total number of parameters is {}'.format(num_of_params))
        # self.net.eval()
        # loss, acc = self._validate(criterion, test_loader, 0, device)
        # wandb.log({'test/loss': loss, 'test/acc': acc})

    def _validate(self, criterion, val_loader, step, device):
        pass
        # loss_meter = pth_utils.AverageMeter()
        # p = []
        # gt = []
        # for samples in val_loader:
        #     inputs = samples[0]
        #     batch_size = inputs.shape[0]
        #     in_dim = inputs.shape[2] * inputs.shape[3]
        #     inputs_vec = inputs.reshape((batch_size, in_dim))
        #     labels = samples[1]
        #     inputs_vec = inputs_vec.to(device)
        #     labels = labels.to(device)
        #
        #     # forward
        #     outputs = self.net(inputs_vec)
        #     loss = criterion(outputs, labels)
        #
        #     probability_outputs = torch.softmax(outputs, dim=1)
        #     predicted_classes = torch.argmax(probability_outputs, dim=1)
        #     p.extend(predicted_classes.tolist())
        #     gt.extend(labels.tolist())
        #     loss_meter.update(loss.item(), batch_size)
        #
        # p = np.array(p, dtype=np.int)
        # gt = np.array(gt, dtype=np.int)
        # acc = accuracy(p, gt)
        # val_loss = loss_meter.avg
        # return val_loss, acc

    def inference(self):
        pass
