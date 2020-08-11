import torch
import wandb
import os


def save_model(model_dict, name, upload_to_wandb=False):
    with torch.no_grad():
        model_name = '{}-{}.pth'.format(name, model_dict['step'])
        checkpoint_file = os.path.join(wandb.run.dir, model_name)
        torch.save(model_dict, checkpoint_file)
        if upload_to_wandb:
            wandb.save(model_name)
        return checkpoint_file


def restore_model(file, storage='local', encoding='utf-8'):
    if storage == 'wandb':
        parts = file.split('/')
        wandb_path = '/'.join(parts[:-1])
        wandb_file = parts[-1]
        restore_file = wandb.restore(wandb_file, run_path=wandb_path)
        checkpoint = torch.load(restore_file.name, encoding=encoding)
    elif storage == 'local':  # local storage
        checkpoint = torch.load(file, encoding=encoding)
    else:
        print('Unknown storage type')
        checkpoint = None

    return checkpoint


def calc_sparsity_percentage(W, eps=1e-5):
    n = W.shape[0] * W.shape[1]
    i = torch.sum(torch.abs(W) < eps)
    return i.item() / n
