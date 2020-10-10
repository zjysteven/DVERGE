import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import arguments, utils
from models.ensemble import Ensemble
from distillation import Linf_PGD


class ADP_Trainer():
    def __init__(self, models, trainloader, testloader,
                 writer, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_root = save_root

        self.log_offset = 1e-20
        self.det_offset = 1e-6
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']

        params = []
        for m in self.models:
            params += list(m.parameters())
        self.optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-4, eps=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=kwargs['sch_intervals'], gamma=kwargs['lr_gamma'])
        
        self.criterion = nn.CrossEntropyLoss()

        self.plus_adv = kwargs['plus_adv']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['adv_eps'], 
                               'alpha': kwargs['adv_alpha'],
                               'steps': kwargs['adv_steps'],
                               'is_targeted': False,
                               'rand_start': True
                              }
    
    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1,self.epochs+1)), total=self.epochs, desc='Epoch',
                        leave=False, position=1)
        return iterator

    def get_batch_iterator(self):
        iterator = tqdm(self.trainloader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            self.test(epoch)
            self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = 0
        ce_losses = 0
        ee_losses = 0
        det_losses = 0
        adv_losses = 0
        
        batch_iter = self.get_batch_iterator()
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            inputs, targets = inputs.cuda(), targets.cuda()

            if self.plus_adv:
                ensemble = Ensemble(self.models)
                adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)

            # one-hot label
            num_classes = 10
            y_true = torch.zeros(inputs.size(0), num_classes).cuda()
            y_true.scatter_(1, targets.view(-1,1), 1)
    
            ce_loss = 0
            adv_loss = 0
            mask_non_y_pred = []
            ensemble_probs = 0
            for i, m in enumerate(self.models):
                outputs = m(inputs)
                ce_loss += self.criterion(outputs, targets)

                # for log_det
                y_pred = F.softmax(outputs, dim=-1)
                bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true)) # batch_size X (num_class X num_models), 2-D
                mask_non_y_pred.append(torch.masked_select(y_pred, bool_R_y_true).reshape(-1, num_classes-1)) # batch_size X (num_class-1) X num_models, 1-D

                # for ensemble entropy
                ensemble_probs += y_pred

                if self.plus_adv:
                    # for adv loss
                    adv_loss += self.criterion(m(adv_inputs), targets)

            ensemble_probs = ensemble_probs / len(self.models)
            ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + self.log_offset)), dim=-1).mean()

            mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
            assert mask_non_y_pred.shape == (inputs.size(0), len(self.models), num_classes-1)
            mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1, keepdim=True) # batch_size X num_model X (num_class-1), 3-D
            matrix = torch.matmul(mask_non_y_pred, mask_non_y_pred.permute(0, 2, 1)) # batch_size X num_model X num_model, 3-D
            log_det = torch.logdet(matrix+self.det_offset*torch.eye(len(self.models), device=matrix.device).unsqueeze(0)).mean() # batch_size X 1, 1-D

            loss = ce_loss - self.alpha * ensemble_entropy - self.beta * log_det + adv_loss

            losses += loss.item()
            ce_losses += ce_loss.item()
            ee_losses += ensemble_entropy.item()
            det_losses += -log_det.item()
            if self.plus_adv:
                adv_losses += adv_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()         
        
        self.scheduler.step()

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(
                i=i+1, loss=losses/(batch_idx+1))
        tqdm.write(print_message)

        self.writer.add_scalar('train/ce_loss', ce_losses/len(self.trainloader), epoch)
        self.writer.add_scalar('train/ce_loss', ce_losses/len(self.trainloader), epoch)
        self.writer.add_scalar('train/ee_loss', ee_losses/len(self.trainloader), epoch)
        self.writer.add_scalar('train/det_loss', det_losses/len(self.trainloader), epoch)
        self.writer.add_scalar('train/adv_loss', adv_losses/len(self.trainloader), epoch)

    def test(self, epoch):
        for m in self.models:
            m.eval()

        ensemble = Ensemble(self.models)

        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = ensemble(inputs)
                loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        self.writer.add_scalar('test/ensemble_loss', loss/len(self.testloader), epoch)
        self.writer.add_scalar('test/ensemble_acc', 100*correct/total, epoch)

        print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
            loss=loss/len(self.testloader), acc=correct/total)
        tqdm.write(print_message)

    def save(self, epoch):
        state_dict = {}
        for i, m in enumerate(self.models):
            state_dict['model_%d'%i] = m.state_dict()
        torch.save(state_dict, os.path.join(self.save_root, 'epoch_%d.pth'%epoch))


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 ADP Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.adp_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    save_root = os.path.join('checkpoints', 'adp', 'seed_{:d}'.format(args.seed), '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth))
    if args.plus_adv:
        save_root += '_plus_adv'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # set up random seed
    torch.manual_seed(args.seed)

    # initialize models
    models = utils.get_models(args, train=True, as_ensemble=False, model_file=None)

    # get data loaders
    trainloader, testloader = utils.get_loaders(args)

    # train the ensemble
    trainer = ADP_Trainer(models, trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
