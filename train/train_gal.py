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
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

import arguments, utils
from models.ensemble import Ensemble
from distillation import Linf_PGD


class GAL_Trainer():
    def __init__(self, models, trainloader, testloader,
                 writer, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_root = save_root

        self.coeff = kwargs['lambda']
        self.log_offset = 1e-20

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
        coh_losses = 0
        adv_losses = 0
        
        batch_iter = self.get_batch_iterator()
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs.requires_grad = True

            if self.plus_adv:
                ensemble = Ensemble(self.models)
                adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)

            ce_loss = 0
            adv_loss = 0
            grads = []
            for i, m in enumerate(self.models):
                # for coherence loss
                outputs = m(inputs)
                loss = self.criterion(outputs, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)

                # for standard loss
                ce_loss += self.criterion(m(inputs.clone().detach()), targets)

                if self.plus_adv:
                    # for adv loss
                    adv_loss += self.criterion(m(adv_inputs), targets)

            cos_sim = []
            for i in range(len(self.models)):
                for j in range(i+1, len(self.models)):
                    cos_sim.append(F.cosine_similarity(grads[i], grads[j], dim=-1))
            
            cos_sim = torch.stack(cos_sim, dim=-1)
            assert cos_sim.shape == (inputs.size(0), (len(self.models)*(len(self.models)-1))//2)
            coh_loss = torch.log(cos_sim.exp().sum(dim=-1)+self.log_offset).mean()
            
            loss = ce_loss / len(self.models) + self.coeff * coh_loss + adv_loss / len(self.models)

            losses += loss.item()
            ce_losses += ce_loss.item()
            coh_losses += coh_loss.item()
            if self.plus_adv:
                adv_losses += adv_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()         
        
        self.scheduler.step()

        print_message = 'Epoch [%3d] | ce_loss: %.4f\tcoh_loss: %.4f\tadv_loss: %.4f' % (epoch, 
            ce_losses/(batch_idx+1), coh_losses/(batch_idx+1), adv_losses/(batch_idx+1))
        tqdm.write(print_message)

        self.writer.add_scalar('train/ce_loss', ce_losses/len(self.trainloader), epoch)
        self.writer.add_scalar('train/coh_loss', coh_losses/len(self.trainloader), epoch)
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
    parser = argparse.ArgumentParser(description='CIFAR10 GAL Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.gal_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    save_root = os.path.join('checkpoints', 'gal', 'seed_{:d}'.format(args.seed), '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth))
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
    random.seed(args.seed)

    # initialize models
    models = utils.get_models(args, train=True, as_ensemble=False, model_file=None, leaky_relu=True)

    # get data loaders
    trainloader, testloader = utils.get_loaders(args, add_gaussian=True)

    # train the ensemble
    trainer = GAL_Trainer(models, trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
