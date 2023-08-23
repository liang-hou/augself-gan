# DiffAugment for BigGAN (CIFAR)

This repo is implemented upon the [BigGAN-PyTorch repo](https://github.com/ajbrock/BigGAN-PyTorch). The main dependencies are:

- PyTorch version >= 1.0.1. Code has been tested with PyTorch 1.4.0.

- TensorFlow 1.14 or 1.15 with GPU support (for IS and FID calculation).

## Training

We provide a complete set of training scripts in the `scripts` folder to facilitate replicating our results. For example, the following command will run *AugSelf-BigGAN(+)* on CIFAR-10 with 10% training data:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/augself-biggan-cifar10-0.1.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/augself+biggan-cifar10-0.1.sh
```
