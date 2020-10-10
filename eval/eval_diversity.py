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
from distillation import Linf_distillation


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of Diversity of Ensembles', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.diversity_eval_args(parser)
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
    testloader = utils.get_testloader(args, batch_size=100)
    
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
    # use a very small batch size so that we can sample different layers multiple times
    subset_loader = utils.get_testloader(args, batch_size=10, shuffle=True, subset_idx=subset_idx)

    eps_list = [0.07]
    steps = 10

    rob = {}
    rob['steps'] = steps
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    for eps in tqdm(eps_list, desc='eps', leave=False, position=0):
        losses = torch.zeros((len(models), len(models), args.subset_num))

        torch.manual_seed(0)
        loader = utils.DistillationLoader(subset_loader, subset_loader)
        test_iter = tqdm(loader, desc='Batch', leave=False, position=1)

        random.seed(0)
        total = 0
        for batch_idx, (si, sl, ti, tl) in enumerate(test_iter):
            si, sl = si.cuda(), sl.cuda()
            ti, tl = ti.cuda(), tl.cuda()

            layer = random.randint(1, args.depth)

            adv_list = []
            for i, m in enumerate(models):
                adv = Linf_distillation(m, si, ti, eps, eps/steps, steps, layer, before_relu=True, mu=1, momentum=True, rand_start=False)
                adv_list.append(adv)

            with torch.no_grad():
                for i, adv in enumerate(adv_list):
                    for j, m in enumerate(models):
                        if j == i:
                            outputs = m(si)
                            _, pred = outputs.max(1)
                            assert pred.eq(sl).all()

                        outputs = m(adv)
                        loss = criterion(outputs, tl)

                        losses[i, j, total:total+si.size(0)] = loss
                
            total += si.size(0)

        losses_np = torch.mean(losses, dim=-1).numpy()

        tqdm.write("eps: {:.2f}".format(eps))

        for i in range(len(models)):
            message = ''
            for j in range(len(models)):
                message += '\t{}: {:.3f}'.format(j, losses_np[i, j])
            tqdm.write(message)
        
        rob[str(eps)] = losses_np
    
    # save to file
    if args.save_to_file:
        output_root = os.path.join('results', 'diversity', train_alg, train_seed)
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_filename = args.model_file.split('/')[-2]
        output = os.path.join(output_root, '.'.join((output_filename, 'pkl')))

        with open(output, 'wb') as f:
            pickle.dump(rob, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
