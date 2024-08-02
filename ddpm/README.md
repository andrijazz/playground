# DDPM
Simple implementation of Denoising Diffusion Probabilistic Models.

### Setup

```bash
# install dependencies
conda create -f environment.yml ddpm
conda activate ddpm
pip install -r requirements.txt

# setup environment variables
export DATASETS=<path to datasets>
export LOG=<path to log dir>

# download datasets
https://github.com/Ryota-Kawamura/How-Diffusion-Models-Work/blob/main/sprites_1788_16x16.npy
https://github.com/Ryota-Kawamura/How-Diffusion-Models-Work/blob/main/sprite_labels_nc_1788_16x16.npy
https://www.kaggle.com/datasets/therealcyberlord/50k-celeba-dataset-64x64?resource=download
```
### Usage

```bash
# train model
python ddpm.py
# test model
python ddpm.py --mode test
```

### References
| Reference                                                                                     | Description                                              |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [How diffusion models work (deeplearning.ai)](https://learn.deeplearning.ai/diffusion-models) | Nice introduction to diffusion models with toy example. |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)                  | Paper that introduces diffusion models.                  |
| [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)                       | Improved version of sampling algorithm.                  |
| [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)         | Making DDPM work in practice.                            |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)         | Classifier guidance.                                     |
| [UNet](https://arxiv.org/pdf/1505.04597.pdf)                                                  |                                                          |
