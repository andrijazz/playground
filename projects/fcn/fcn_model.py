from __future__ import absolute_import, division

import os

import numpy as np
import tensorflow as tf
import wandb

import core.factory as factory

from projects.fcn.fcn_datasets import FCNDataset
import projects.utils.utils as utils
from core.base_model import BaseModel
from core.logger import get_logger
import projects.utils.tf_utils as tf_utils
from projects.utils.metrics import micro_iou

logger = get_logger()

# enable dryrun to turn of wandb syncing completely
# os.environ['WANDB_MODE'] = 'dryrun'
# prevent wandb syncing checkpoints except best model results
os.environ['WANDB_IGNORE_GLOBS'] = 'val_checkpoint*,checkpoint*'


class FCNModel(BaseModel):
    """
    Model for fcn net
    """
    def __init__(self, config):
        super().__init__(config)

        # construct the dataset
        self.dataset = FCNDataset(self.config.TRAIN_DATASET)

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

    def save(self, checkpoint_name, step, upload_to_wandb=False):
        checkpoint = os.path.join(wandb.run.dir, '{}-{}'.format(checkpoint_name, step))
        self.net.save_weights(checkpoint)
        if upload_to_wandb:
            wandb.save(checkpoint)
        return checkpoint

    def train(self):
        wandb.init(project=os.getenv('PROJECT'), dir=os.getenv('LOG'), config=self.config, reinit=True)

        if self.config.TRAIN_RESTORE_FILE:
            self.restore(self.config.TRAIN_RESTORE_FILE, self.config.TRAIN_RESTORE_STORAGE)

        train_dataset, val_dataset = self.dataset.create_train_and_val_datasets()
        train_dataset_len = tf.data.experimental.cardinality(train_dataset).numpy()
        val_dataset_len = tf.data.experimental.cardinality(val_dataset).numpy()

        # log dataset sizes
        logger.info('Training dataset size = {}'.format(train_dataset_len))
        logger.info('Validation dataset size = {}'.format(val_dataset_len))

        # shuffle datasets
        train_dataset = train_dataset.shuffle(train_dataset_len)
        train_dataset = train_dataset.batch(self.config.TRAIN_BATCH_SIZE)
        val_dataset = val_dataset.shuffle(val_dataset_len)
        val_dataset = val_dataset.batch(self.config.VAL_BATCH_SIZE)

        # instantiate an optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.TRAIN_LR)

        # instantiate a loss function.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        step = 0

        # init best checkpoint info
        best_checkpoint_info = {'step': step, 'loss': np.inf, 'iou': 0, 'checkpoint': ''}

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
                    argmax_logits = tf.argmax(logits, axis=-1)
                    rgb_batch = tf_utils.idx_to_rgb_batch_tf(argmax_logits, self.dataset.idx_to_color)

                    # log scalars
                    wandb.log({"train/loss": loss_value}, step=step)

                    # log images
                    # current batch_size
                    batch_size = x_batch_train.shape[0]
                    # choose random image in a batch
                    idx = np.random.choice(batch_size)
                    image = x_batch_train[idx]
                    gt_image = y_batch_train[idx]
                    y_pred = rgb_batch[idx]
                    wandb.log({"train/image": wandb.Image(image)}, step=step)
                    wandb.log({"train/gt_image": wandb.Image(gt_image)}, step=step)
                    wandb.log({"train/y_pred": wandb.Image(y_pred)}, step=step)

                if self.config.TRAIN_VAL_FREQ != -1 and step % self.config.TRAIN_VAL_FREQ == 0:
                    val_loss, val_iou = self._validate(val_dataset, loss_fn, step)
                    # log scalars
                    wandb.log({"val/loss": val_loss}, step=step)
                    wandb.log({"val/iou": val_iou}, step=step)

                    # if val_loss > best_checkpoint_info['loss'] and epoch > self.config.TRAIN_START_SAVING_AFTER_EPOCH:
                    best_checkpoint_info['loss'] = val_loss
                    best_checkpoint_info['iou'] = val_iou
                    best_checkpoint_info['step'] = step
                    checkpoint = self.save('val_checkpoint', step, upload_to_wandb=False)
                    best_checkpoint_info['checkpoint'] = checkpoint

                if self.config.TRAIN_SAVE_MODEL_FREQ != -1 and step % self.config.TRAIN_SAVE_MODEL_FREQ == 0:
                    self.save('checkpoint', step)

                step += 1

        # # restore best checkpoint state
        self.restore(best_checkpoint_info['checkpoint'], storage='local')
        # save best model as {model_name} and upload it to wandb if specified
        model_name = self.config.MODEL.lower()
        self.save(model_name, best_checkpoint_info['step'])
        return best_checkpoint_info['iou']

    def test(self):
        pass

    def _validate(self, val_dataset, loss_fn, step):
        iou_meter = utils.AverageMeter()
        loss_meter = utils.AverageMeter()
        log_image = True
        for (x_batch_val, y_batch_val, y_batch_val_idx) in val_dataset:
            logits = self.net(x_batch_val, training=True)

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_val_idx, logits)

            argmax_logits = tf.argmax(logits, axis=-1)
            batch_size = x_batch_val.shape[0]

            # log image
            if log_image:
                rgb_batch = tf_utils.idx_to_rgb_batch_tf(argmax_logits, self.dataset.idx_to_color)

                # choose random image in a batch
                idx = np.random.choice(batch_size)
                image = x_batch_val[idx]
                gt_image = y_batch_val[idx]
                y_pred = rgb_batch[idx]
                wandb.log({"val/image": wandb.Image(image)}, step=step)
                wandb.log({"val/gt_image": wandb.Image(gt_image)}, step=step)
                wandb.log({"val/y_pred": wandb.Image(y_pred)}, step=step)
                log_image = False

            loss_meter.update(loss_value, batch_size)
            # iou_per_class = micro_iou(rgb_batch, y_batch_val, kitti.idx_to_color)
            # iou = np.mean(iou_per_class)
            # iou_meter.update(iou, batch_size)

        val_loss = loss_meter.avg
        val_iou = iou_meter.avg
        return val_loss, val_iou

    def inference(self, x):
        if len(x.shape) == 3:  # single image
            h = x.shape[0]
            w = x.shape[1]
            ch = x.shape[2]
            # reshape to batch (1, h, w, ch)
            x = tf.reshape(x, [-1, h, w, ch])
            logits = self.net.call(x)
            x = tf.argmax(logits, axis=-1)
            # reshape back to single image
            x = x[0]
            x = tf_utils.idx_to_rgb_tf(x, self.dataset.idx_to_color)
            return x
        elif len(x.shape) == 4:  # batch
            logits = self.net.call(x)
            x = tf.argmax(logits, axis=-1)
            x = tf_utils.idx_to_rgb_batch_tf(x, self.dataset.idx_to_color)
            return x
        else:
            exit('Unsupported tensor shape for fcn inference')

