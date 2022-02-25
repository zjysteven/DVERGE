# DVERGE
This repository contains code for reproducing our NeurIPS 2020 **Oral** paper ["DVERGE: Diversifying Vulnerabilities for Enhanced Robust Generation of Ensembles"](https://papers.nips.cc/paper/2020/hash/3ad7c2ebb96fcba7cda0cf54a2e802f5-Abstract.html).

# Dependencies
Create the conda environment called `dverge` containing all the dependencies by running
```
conda env create -f environment.yml
```
We were using PyTorch 1.4.0 for all the experiments. You may want to install other versions of PyTorch according to the cuda version of your computer/server.
The code is run and tested on a single TITAN Xp GPU. Running on multiple GPUs with parallelism may need adjustments.

# Data and pre-trained models
The pre-trained models and generated black-box transfer adversarial examples can be accessed via [this link](https://drive.google.com/drive/folders/1i96Bk_bCWXhb7afSNp1t3woNjO1kAMDH?usp=sharing). Specifically, the pre-trained models are stored in the folder named `checkpoints`. Download and put `checkpoints` under this repo.

The black-box transfer adversarial examples (refer to the paper for more details) are stored in `transfer_adv_examples.zip`. Make a folder named `data` under this repo. Download the zip file, unzip it, and put the extracted folder `transfer_adv_examples/` under `data/`. Then one can evaluate the black-box transfer robustness of ensembles.

# Usage
Examples of training and evaluation scripts can be found in `scripts/training.sh` and `scripts/evaluation.sh`.

Note that for now we extract models' intermediate features in a very naive way which may only support the ResNet20 architecture. One can implement a more robust feature extraction with the help of `forward hook` of Pytorch.

Also, you may observe a high variation in results when training DVERGE, which we suspect is due to the random layer sampling for distillation. Please refer to **Appendix C.5** of the paper for a discussion on the layer effects.

# Decision region plot
We have been receiving many questions regarding the decision region plot in Figure 1. To understand how it works, a neat working example can be found in the "What is happening with these robust models?" section in [this fantastic tutorial](https://adversarial-ml-tutorial.org/adversarial_training/). Our code is adapted from that example, and the only difference is that while they plot the loss, we plot the model's decision/predicted class. Our code can be found [here](https://drive.google.com/file/d/1KNoQGTXm3g_RBwE0a6IkrlSks4Wez_tN/view). It is pretty messy, yet the essential part starts from line 177. When plotting Figure 1, we use `args.steps=1000` and `args.vmax=0.1`, which means that we are perturbing along each direction by a maximum of distance of `0.1`, and along each direction we sample `1000` perturbations and record the model's decision on each of the corresponding perturbed sample. So totally we sample `1000*1000` data points to make each of the plot in Figure 1.


# Reference
If you find our paper/this repo useful for your research, please consider citing our work.
```
@article{yang2020dverge,
  title={DVERGE: Diversifying Vulnerabilities for Enhanced Robust Generation of Ensembles},
  author={Yang, Huanrui and Zhang, Jingyang and Dong, Hongliang and Inkawhich, Nathan and Gardner, Andrew and Touchet, Andrew and Wilkes, Wesley and Berry, Heath and Li, Hai},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

# Acknowledgement
The training code of [ADP](https://arxiv.org/pdf/1901.08846.pdf) (Adaptive Diversity Promoting Regularizer) is adapted from [the official repo](https://github.com/P2333/Adaptive-Diversity-Promoting), which is originally written in TensorFlow and we turned it into Pytorch here.
