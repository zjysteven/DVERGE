import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

import arguments, utils
from models.ensemble import Ensemble


def test(model, datafolder, return_acc=False):
    inputs = torch.load(os.path.join(datafolder, 'inputs.pt')).cpu()
    labels = torch.load(os.path.join(datafolder, 'labels.pt')).cpu()
    
    testset = TensorDataset(inputs, labels)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    
    correct = []
    with torch.no_grad():
        for (x, y) in testloader:
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            _, preds = outputs.max(1)
            correct.append(preds.eq(y))
    correct = torch.cat(correct, dim=0)
    if return_acc:
        return 100.*correct.sum().item()/len(testset)
    else:
        return correct


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of Black-box Transfer Robustness of Ensembles', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.bbox_eval_args(parser)
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

    train_seed = args.model_file.split('/')[-3]
    train_alg = args.model_file.split('/')[-4]

    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    loss_fn_list = ['xent', 'cw']
    surrogate_model_list = ['{:s}{:d}'.format(args.which_ensemble, i) for i in [3, 5, 8]]
    method_list = ['mdi2_0.5_steps_{:d}'.format(args.steps), 'sgm_0.2_steps_{:d}'.format(args.steps)]
    index_list = ['{:s}_{:s}_mpgd'.format(a, b) for a in surrogate_model_list for b in loss_fn_list]
    index_list += ['{:s}_{:s}_{:s}'.format(a, b, c) for a in surrogate_model_list for b in loss_fn_list for c in method_list]
    index_list.append('all')

    random_start = 3
    input_list = ['from_{:s}_{:s}_mpgd_steps_{:d}'.format(a, b, args.steps) for a in surrogate_model_list for b in loss_fn_list]
    input_list += ['from_{:s}_{:s}_{:s}'.format(a, b, c) for a in surrogate_model_list for b in loss_fn_list for c in method_list]

    rob = {}
    rob['source'] = index_list
    acc_list = [[] for _ in range(len(eps_list))]

    data_root = os.path.join(args.data_dir, args.folder)

    # clean acc
    input_folder = os.path.join(data_root, 'clean')
    clean_acc = test(ensemble, input_folder, return_acc=True)
    clean_acc_list = [clean_acc for _ in range(len(input_list)+1)]
    rob['clean'] = clean_acc_list

    # transfer attacks    
    for i, eps in enumerate(tqdm(eps_list, desc='eps', leave=True, position=0)):
        input_folder = os.path.join(data_root, 'eps_{:.2f}'.format(eps))
        correct_over_input = []

        for input_adv in tqdm(input_list, desc='source', leave=False, position=1):
            if 'mpgd' in input_adv:
                correct_over_rs = []

                for rs in tqdm(range(random_start), desc='Random Start', leave=False, position=2):
                    datafolder = os.path.join(input_folder, '_'.join((input_adv, str(rs))))
                    correct_over_rs.append(test(ensemble, datafolder))

                correct_over_rs = torch.stack(correct_over_rs, dim=-1).all(dim=-1)
                acc_list[i].append(100.*correct_over_rs.sum().item()/len(correct_over_rs))
                correct_over_input.append(correct_over_rs)

                tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
                    clean_acc, eps, input_adv, 100.*correct_over_rs.sum().item()/len(correct_over_rs)
                ))
            else:
                datafolder = os.path.join(input_folder, input_adv)
                correct = test(ensemble, datafolder)
                acc_list[i].append(100.*correct.sum().item()/len(correct))
                correct_over_input.append(correct)

                tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
                    clean_acc, eps, input_adv, 100.*correct.sum().item()/len(correct)
                ))

        correct_over_input = torch.stack(correct_over_input, dim=-1).all(dim=-1)
        acc_list[i].append(100.*correct_over_input.sum().item()/len(correct_over_input))
    
        tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer acc: {:.2f}%'.format(
            clean_acc, eps, 100.*correct_over_input.sum().item()/len(correct_over_input)
        ))
    
        rob[str(eps)] = acc_list[i]

    # save to file
    if args.save_to_csv:
        output_root = os.path.join('results', 'bbox', train_alg, train_seed)
        
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_filename = args.model_file.split('/')[-2]
        output = os.path.join(output_root, '.'.join((output_filename, 'csv')))

        df = pd.DataFrame(rob)
        if args.append_out and os.path.isfile(output):
            with open(output, 'a') as f:
                f.write('\n')
            df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
        else:
            df.to_csv(output, sep=',', index=False, float_format='%.2f')
    
    
if __name__ == '__main__':
    main()
