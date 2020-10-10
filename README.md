# DVERGE
This repository contains code for reproducing our NeurIPS 2020 paper "DVERGE: Diversifying Vulnerabilities for Enhanced Robust Generation of Ensembles".

# Dependencies
Create the conda environment called `ensemble` containing all the dependencies by running
```
conda env create -f environment.yml
```
The code is run and tested on a single TITAN Xp GPU. Running with multiple GPUs with parallelism may need adjustments.

# Data and pre-trained models
The pre-trained DVERGE models (with ResNet-20s as the architecture) are already tracked in `checkpoints/dverge`.


The black-box transfer adversarial examples are uploaded via [this link](https://drive.google.com/file/d/1hL3SK9QWZVADHsCR8CcGvbopOvPe3Ylk/view?usp=sharing). Please refer to the paper for more details. Make a folder named `data` under this repo. Download the transfer adversarial examples, unzip the zip file, and put the extracted folder `transfer_adv_examples/` under `data/`. Then one can evaluate the black-box transfer robustness of ensembles.

# Usage
Examples of training and evaluation scripts can be found in `scripts/training.sh` and `scripts/evaluation.sh`.

Note that for now we extract models' intermediate features in a very naive way. One can implement a more robust feature extraction with the help of `forward hook` of Pytorch.

# Acknowledgement
The training code of [ADP](https://arxiv.org/pdf/1901.08846.pdf) (Adaptive Diversity Promoting Regularizer) is adapted from [the official repo](https://github.com/P2333/Adaptive-Diversity-Promoting), which is originally written in TensorFlow and we turned it into Pytorch here.
