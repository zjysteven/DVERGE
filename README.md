# DVERGE
This repository contains code for reproducing our NeurIPS 2020 paper "DVERGE: Diversifying Vulnerabilities for Enhanced Robust Generation of Ensembles".

# Dependencies
Create the conda environment called `dverge` containing all the dependencies by running
```
conda env create -f environment.yml
```
The code is run and tested on a single TITAN Xp GPU. Running on multiple GPUs with parallelism may need adjustments.

# Data and pre-trained models
The pre-trained models and generated black-box transfer adversarial examples can be accessed via [this link](https://drive.google.com/drive/folders/1i96Bk_bCWXhb7afSNp1t3woNjO1kAMDH?usp=sharing). Specifically, the pre-trained models are stored in the folder named `checkpoints`. Download and put `checkpoints` under this repo.

The black-box transfer adversarial examples (refer to the paper for more details) are stored in `transfer_adv_examples.zip`. Make a folder named `data` under this repo. Download the zip file, unzip the it, and put the extracted folder `transfer_adv_examples/` under `data/`. Then one can evaluate the black-box transfer robustness of ensembles.

# Usage
Examples of training and evaluation scripts can be found in `scripts/training.sh` and `scripts/evaluation.sh`.

Note that for now we extract models' intermediate features in a very naive way which may only support the ResNet20 architecture. One can implement a more robust feature extraction with the help of `forward hook` of Pytorch.

Also, you may observe a high variation in results when training DVERGE, which we suspect is due to the random layer sampling for distillation. Please refer to **Appendix C.5** of the paper for a discussion on the layer effects.

# Acknowledgement
The training code of [ADP](https://arxiv.org/pdf/1901.08846.pdf) (Adaptive Diversity Promoting Regularizer) is adapted from [the official repo](https://github.com/P2333/Adaptive-Diversity-Promoting), which is originally written in TensorFlow and we turned it into Pytorch here.
