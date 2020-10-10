import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import argparse, random
from tqdm import tqdm
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from advertorch.utils import to_one_hot

import arguments, utils
from models.ensemble import Ensemble
from distillation import Linf_PGD


# https://github.com/BorealisAI/advertorch/blob/master/advertorch/utils.py
class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """
    def __init__(self, conf=50.):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.conf).sum()
        return loss


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of White-box Robustness of Ensembles with Advertorch', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.wbox_eval_args(parser)
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

    # get data loaders
    total_sample_num = 10000
    if args.subset_num:
        random.seed(0)
        subset_idx = random.sample(range(total_sample_num), args.subset_num)
        testloader = utils.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_idx)
    else:
        testloader = utils.get_testloader(args, batch_size=200, shuffle=False)

    loss_fn = nn.CrossEntropyLoss() if args.loss_fn == 'xent' else CarliniWagnerLoss(conf=args.cw_conf)

    rob = {}
    rob['sample_num'] = args.subset_num if args.subset_num else total_sample_num
    rob['loss_fn'] = 'xent' if args.loss_fn == 'xent' else 'cw_{:.1f}'.format(args.cw_conf)

    train_seed = args.model_file.split('/')[-3]
    train_alg = args.model_file.split('/')[-4]

    if args.convergence_check:
        eps = 0.01
        steps_list = [50, 500, 1000]
        random_start = 1

        rob['random_start'] = random_start
        rob['eps'] = eps

        # FGSM
        test_iter = tqdm(testloader, desc='FGSM', leave=False, position=0)
        adversary = GradientSignAttack(
            ensemble, loss_fn=nn.CrossEntropyLoss(), eps=eps, 
            clip_min=0., clip_max=1., targeted=False)
        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
        print("Accuracy: {:.2f}%, FGSM Accuracy: {:.2f}%".format(
            100. * (label == pred).sum().item() / len(label),
            100. * (label == advpred).sum().item() / len(label)))
        rob['clean'] = 100. * (label == pred).sum().item() / len(label)
        rob['fgsm'] = 100. * (label == advpred).sum().item() / len(label)
        
        for steps in tqdm(steps_list, desc='PGD steps', leave=False, position=0):
            correct_or_not = []

            for i in tqdm(range(random_start), desc='Random Start', leave=False, position=1):
                torch.manual_seed(i)
                test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

                adversary = LinfPGDAttack(
                    ensemble, loss_fn=loss_fn, eps=eps,
                    nb_iter=steps, eps_iter=eps/5, rand_init=True, clip_min=0., clip_max=1.,
                    targeted=False)
                
                _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda") 
                correct_or_not.append(label == advpred)
            
            correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

            tqdm.write("Accuracy: {:.2f}%, steps: {:d}, PGD Accuracy: {:.2f}%".format(
                100. * (label == pred).sum().item() / len(label),
                steps,
                100. * correct_or_not.sum().item() / len(label)))
            
            rob[str(steps)] = 100. * correct_or_not.sum().item() / len(label)
        
        # save to file
        if args.save_to_csv:
            output_root = os.path.join('results', 'wbox', train_alg, train_seed, 'convergence_check')
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            output_filename = args.model_file.split('/')[-2]
            output = os.path.join(output_root, '.'.join((output_filename, 'csv')))

            df = pd.DataFrame(rob, index=[0])
            if args.append_out and os.path.isfile(output):
                with open(output, 'a') as f:
                    f.write('\n')
                df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
            else:
                df.to_csv(output, sep=',', index=False, float_format='%.2f')
    else:
        eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]

        rob['random_start'] = args.random_start
        rob['steps'] = args.steps
        
        for eps in tqdm(eps_list, desc='PGD eps', leave=True, position=0):            
            correct_or_not = []

            for i in tqdm(range(args.random_start), desc='Random Start', leave=False, position=1):
                torch.manual_seed(i)
                test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

                adversary = LinfPGDAttack(
                    ensemble, loss_fn=loss_fn, eps=eps,
                    nb_iter=args.steps, eps_iter=eps/5, rand_init=True, clip_min=0., clip_max=1.,
                    targeted=False)
                
                _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")

                correct_or_not.append(label == advpred)
            
            correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

            tqdm.write("Accuracy: {:.2f}%, eps: {:.2f}, PGD Accuracy: {:.2f}%".format(
                100. * (label == pred).sum().item() / len(label),
                eps,
                100. * correct_or_not.sum().item() / len(label)))
            
            rob['clean'] = 100. * (label == pred).sum().item() / len(label)
            rob[str(eps)] = 100. * correct_or_not.sum().item() / len(label)
        
        # save to file
        if args.save_to_csv:
            output_root = os.path.join('results', 'wbox', train_alg, train_seed)
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            output_filename = args.model_file.split('/')[-2]
            output = os.path.join(output_root, '.'.join((output_filename, 'csv')))

            df = pd.DataFrame(rob, index=[0])
            if args.append_out and os.path.isfile(output):
                with open(output, 'a') as f:
                    f.write('\n')
                df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
            else:
                df.to_csv(output, sep=',', index=False, float_format='%.2f')


if __name__ == '__main__':
    main()
