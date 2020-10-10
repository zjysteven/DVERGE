import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import argparse, random, pickle
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from advertorch.utils import predict_from_logits

import arguments, utils
from models.ensemble import Ensemble


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of Transferability within Ensembles', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.transf_eval_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # load models
    if 'gal' in args.model_file:
        leaky_relu = True
    else:
        leaky_relu = False
    ensemble = utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)
    models = ensemble.models

    train_seed = args.model_file.split('/')[-3]
    train_alg = args.model_file.split('/')[-4]

    # get data loaders
    testloader = utils.get_testloader(args, batch_size=args.batch_size)
    
    # pick out samples that are correctly classified by all submodels
    correct = []
    for m in models:
        correct_m = []
        for (x, y) in testloader:
            x, y = x.cuda(), y.cuda()

            outputs = m(x)
            _, pred = outputs.max(1)
            correct_m.append(pred.eq(y))
        correct_m = torch.cat(correct_m)
        correct.append(correct_m)
    correct = torch.stack(correct, dim=-1).all(-1)
    correct_idx = correct.nonzero().squeeze(-1)

    random.seed(0)
    subset_idx = correct_idx[random.sample(range(correct_idx.size(0)), args.subset_num)].cpu()
    subset_loader = utils.get_testloader(args, batch_size=args.batch_size, shuffle=False, subset_idx=subset_idx)

    # PGD
    eps_list = [0.03]
    random_start = args.random_start
    steps = args.steps

    rob = {}
    rob['random_start'] = args.random_start
    rob['steps'] = args.steps
    
    for eps in tqdm(eps_list, desc='PGD eps', leave=False, position=0):
        correct_or_not_rs = torch.zeros((len(models), len(models)+1, args.subset_num, random_start), dtype=torch.bool)

        for rs in tqdm(range(random_start), desc='Random Start', leave=False, position=1):
            torch.manual_seed(rs)
            test_iter = tqdm(subset_loader, desc='Batch', leave=False, position=2)

            total = 0
            for (x, y) in test_iter:
                x, y = x.cuda(), y.cuda()

                adv_list = []
                for i, m in enumerate(models):
                    adversary = LinfPGDAttack(
                        m, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                        nb_iter=steps, eps_iter=eps/5, rand_init=True, clip_min=0., clip_max=1.,
                        targeted=False)
            
                    adv = adversary.perturb(x, y)
                    adv_list.append(adv)

                for i, adv in enumerate(adv_list):
                    for j, m in enumerate(models):
                        if j == i:
                            outputs = m(x)
                            _, pred = outputs.max(1)
                            assert pred.eq(y).all()

                        outputs = m(adv)
                        _, pred = outputs.max(1)

                        correct_or_not_rs[i, j, total:total+x.size(0), rs] = pred.eq(y)
                
                    outputs = ensemble(adv)
                    _, pred = outputs.max(1)
                    correct_or_not_rs[i, len(models), total:total+x.size(0), rs] = pred.eq(y)
                
                total += x.size(0)

        correct_or_not_rs = torch.all(correct_or_not_rs, dim=-1)
        asr = np.zeros((len(models), len(models)+1))

        tqdm.write("eps: {:.2f}".format(eps))

        for i in range(len(models)):
            message = ''
            for j in range(len(models)+1):
                message += '\t{}: {:.2%}'.format(j, 1-correct_or_not_rs[i, j, :].sum().item()/args.subset_num)
                asr[i, j] = 1-correct_or_not_rs[i, j, :].sum().item()/args.subset_num
            tqdm.write(message)
        
        rob[str(eps)] = asr
    
    # save to file
    if args.save_to_file:
        output_root = os.path.join('results', 'transferability', train_alg, train_seed)
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_filename = args.model_file.split('/')[-2]
        output = os.path.join(output_root, '.'.join((output_filename, 'pkl')))

        with open(output, 'wb') as f:
            pickle.dump(rob, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
