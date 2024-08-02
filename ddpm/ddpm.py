import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import wandb
from PIL import Image

from unet import ContextUnet


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape

    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)

    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape


def convert_tensor_to_pil(img_tensor):
    tensor = (img_tensor.clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)

    try:
        array = tensor.numpy().astype('uint8')
    except:
        array = tensor.detach().numpy().astype('uint8')

    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)

    im = Image.fromarray(array)
    return im


# construct DDPM noise schedule
def get_ddpm_schedule(timesteps, beta1, beta2, device):
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1
    return a_t, b_t, ab_t


def denoise_add_noise(x, t, pred_noise, z=None, ddpm_schedule=None):
    (a_t, b_t, ab_t) = ddpm_schedule

    # but also adds a little bit of noise back
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


# sampling algorithm for ddpm
@torch.no_grad()
def ddpm_sample(model, ddpm_schedule, batch_size, timesteps=100, resolution=(64, 64), device='cuda'):
    samples = torch.randn(batch_size, 3, resolution[0], resolution[1], device=device)
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        # little extra noise to add back for the next step
        extra_noise = torch.randn_like(samples) if i > 1 else 0
        predicted_noise = model(samples, t)
        samples = denoise_add_noise(samples, i, predicted_noise, extra_noise, ddpm_schedule=ddpm_schedule)
    return samples


def perturb_input(x, t, noise, ddpm_schedule):
    (a_t, b_t, ab_t) = ddpm_schedule
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


def train_epoch(model, ddpm_schedule, dataloader, optimizer, criterion, epoch, config, device):
    model.train()
    pbar = tqdm.tqdm(dataloader)
    for step, x in enumerate(dataloader):
        x = x[0].to(device)
        optimizer.zero_grad()
        # sample random noise
        z = torch.randn_like(x)
        # sample random timestep
        t = torch.randint(1, config['timesteps'] + 1, (x.shape[0],), device=device)
        # NOTE: scale noise by timestep
        x_pert = perturb_input(x, t, z, ddpm_schedule)
        z_pred = model(x_pert, t / config['timesteps'])
        loss = criterion(z_pred, z)
        loss.backward()
        optimizer.step()

        pbar.update()

        if (step + 1) % config['log_freq'] == 0:
            log_dict = {
                'train/loss': loss.item(),
                'train/epoch': epoch,
                'train/real': wandb.Image(x[0].detach().cpu().float()),
            }
            wandb.log(log_dict)

    pbar.close()


def validate(model, ddpm_schedule, config, device):
    model.eval()
    fakes = ddpm_sample(model,
                        ddpm_schedule,
                        batch_size=config['batch_size'],
                        timesteps=config['timesteps'],
                        resolution=config['resolution'],
                        device=device)
    fakes = fakes.detach().cpu().float()
    log_dict = {
        'train/fake': wandb.Image(fakes[0]),
    }
    wandb.log(log_dict)


def save_checkpoint(model, epoch, config):
    checkpoint = {
        'epoch': epoch,
        'config': config,
        'model': model.state_dict(),
    }
    checkpoint_path = os.path.join(wandb.run.dir, f'ddpm_{wandb.run.id}_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)


def train_func():
    config = dict(
        n_feat=64,
        n_cfeat=5,
        timesteps=500,
        beta1=1e-4,
        beta2=0.02,
        batch_size=32,
        resolution=(16, 16),
        learning_rate=1e-3,
        num_epochs=100,
        log_freq=10,
    )

    wandb.init(project='ddpm', dir=os.getenv("LOG"), config=config)
    model = ContextUnet(in_channels=3, n_feat=config['n_feat'], n_cfeat=config['n_cfeat'], height=config['resolution'][0])

    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(size=config['resolution']),
        # torchvision.transforms.CenterCrop(size=config['resolution']),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.ImageFolder(os.path.join(os.getenv('DATASETS'), '50k-celeba'), transform=transform)
    # im_path = os.path.join(os.getenv('DATASETS'), 'sprite', 'sprites_1788_16x16.npy')
    # labels_path = os.path.join(os.getenv('DATASETS'), 'sprite', 'sprite_labels_nc_1788_16x16.npy')
    # dataset = CustomDataset(im_path, labels_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    ddpm_schedule = get_ddpm_schedule(config['timesteps'], config['beta1'], config['beta2'], device=device)
    # training loop
    for epoch in range(config['num_epochs']):
        # NOTE: linearly decay learning rate
        # optimizer.param_groups[0]['lr'] = config['learning_rate'] * (1 - epoch / config['num_epochs'])

        train_epoch(model, ddpm_schedule, dataloader, optimizer, criterion, epoch, config, device)
        validate(model, ddpm_schedule, config, device)
        save_checkpoint(model, epoch, config)

    wandb.finish()


def test():
    config = dict(
        n_feat=64,
        n_cfeat=5,
        timesteps=500,
        beta1=1e-4,
        beta2=0.02,
        batch_size=1,
        resolution=(16, 16),
        learning_rate=1e-3,
        num_epochs=100,
        log_freq=100,
    )

    model = ContextUnet(in_channels=3,
                        n_feat=config['n_feat'],
                        n_cfeat=config['n_cfeat'],
                        height=config['resolution'][0])
    checkpoint_path = os.path.join(os.getenv("MODELS"), 'ddpm', 'ddpm_aaiss0fh_42.pth')
    # checkpoint_path = 'checkpoints/model_trained.pth'
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()
    device = 'cuda'
    model = model.to(device)
    ddpm_schedule = get_ddpm_schedule(timesteps=config['timesteps'],
                                      beta1=config['beta1'],
                                      beta2=config['beta2'],
                                      device=device)
    samples = ddpm_sample(model,
                          ddpm_schedule=ddpm_schedule,
                          batch_size=config['batch_size'],
                          timesteps=config['timesteps'],
                          resolution=config['resolution'],
                          device=device)
    convert_tensor_to_pil(samples[0]).show()


def main(args):
    if args.mode == 'train':
        train_func()
    elif args.mode == 'test':
        test()
    else:
        raise ValueError('Invalid mode!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Script mode.")
    args = parser.parse_args()
    main(args)
