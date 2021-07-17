import torch
import wandb
import os
import torch.utils.data
import time
import gc


def start_timer():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()
    return start_time


def end_timer_and_print(start_time, local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
    return end_time


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# TODO FIX
# checkpoint stays in memory and occupies space
# restoring function should load chkp, load it into model and exit with loaded model
def save_model(model_dict, name, upload_to_wandb=False):
    model_name = '{}-{}.pth'.format(name, model_dict['step'])
    checkpoint_file = os.path.join(wandb.run.dir, model_name)
    torch.save(model_dict, checkpoint_file)
    if upload_to_wandb:
        wandb.save(model_name)


# TODO FIX
# checkpoint stays in memory and occupies space
# restoring function should load chkp, load it into model and exit with loaded model
def restore_model(file, storage='local',  encoding='utf-8'):
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


def normalize(tensor, range, scale_each=False):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if range is not None:
        assert isinstance(range, tuple), \
            "range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, range)
    else:
        norm_range(tensor, range)
    return tensor


# PyTorch Optimization tutorial
# https://www.youtube.com/watch?v=9mS1fIYj1So
def zero_grad(model):
    for p in model.parameters():
        p.grad = None


# https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


# apex
try:
    from apex import amp
except ImportError:
    amp = None


def backward(loss, use_fp16=False, optimizer=None, retain_graph=False, loss_id=0):
    if use_fp16 and amp:
        with amp.scale_loss(loss, optimizer, loss_id) as loss_scaled:
            loss_scaled.backward()
    else:
        loss.backward(retain_graph=retain_graph)


# Grad clipping example in ray
# https://github.com/ray-project/ray/blob/master/python/ray/util/sgd/torch/examples/transformers/transformers_example.py
def optim_step(optimizer, model, gradient_clipping, clip_value):
    grad_n = grad_norm(model)
    if gradient_clipping:
        # grad_n = grad_norm(model)
        if grad_n > clip_value * 100:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            print('clipping grads from {} to {}'.format(grad_n, clip_value))
    optimizer.step()
    return grad_n


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# class FastDataLoader(torch.utils.data.dataloader.DataLoader):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
#         self.iterator = super().__iter__()
#
#     def __len__(self):
#         return len(self.batch_sampler.sampler)
#
#     def __iter__(self):
#         for i in range(len(self)):
#             yield next(self.iterator)

class FastDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    locations = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue

        locations[obj] = obj.device
        obj.data = obj.data.cpu()
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad.data = obj.grad.cpu()

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def calc_sparsity_percentage(W, eps=1e-5):
    n = W.shape[0] * W.shape[1]
    i = torch.sum(torch.abs(W) < eps)
    return i.item() / n