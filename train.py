from __future__ import print_function

import argparse
import csv
import os, logging

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import models
from utils import progress_bar, set_logging_defaults
from datasets import load_dataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="CIFAR_ResNet18", type=str,
                    help='model type (default: CIFAR_ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--num-samples-per-class', default=None, type=int)
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--dataset', default='cifar100', type=str, help='the name for dataset cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67')
parser.add_argument('--dataroot', default='~/data/', type=str, help='data directory')
parser.add_argument('--saveroot', default='./results', type=str, help='save directory')
parser.add_argument('--cls', '-cls', action='store_true', help='adding cls loss')
parser.add_argument('--sam', '-sam', action='store_true', help='adding sam loss')
parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')
parser.add_argument('--lamda2', default=1.0, type=float, help='sam loss weight ratio')
parser.add_argument('--num-iter', default=0, type=int, help='number of iteration per epoch')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
best_acc2 = 0  # best test accuracy
best_val = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

cudnn.benchmark = True

# Data
print('==> Preparing dataset: {}'.format(args.dataset))
if not args.cls:
    if args.sam:
        if args.num_iter > 0:
            trainloader, valloader, testloader = load_dataset(args.dataset, args.dataroot,
                                                              batch_size=args.batch_size,
                                                              num_samples_per_class=args.num_samples_per_class, num_iterations=args.num_iter, sam=True)
        else:
            trainloader, valloader, testloader = load_dataset(args.dataset, args.dataroot,
                                                              batch_size=args.batch_size,
                                                              num_samples_per_class=args.num_samples_per_class, sam=True)
    else:
        if args.num_iter > 0:
            trainloader, valloader, testloader = load_dataset(args.dataset, args.dataroot,
                                                              batch_size=args.batch_size,
                                                              num_samples_per_class=args.num_samples_per_class, num_iterations=args.num_iter)
        else:
            trainloader, valloader, testloader = load_dataset(args.dataset, args.dataroot,
                                                              batch_size=args.batch_size,
                                                              num_samples_per_class=args.num_samples_per_class)
else:
    if args.sam:
        if args.num_iter > 0:
            trainloader, valloader, testloader = load_dataset(args.dataset, args.dataroot, 'pair',
                                                              batch_size=args.batch_size,
                                                              num_samples_per_class=args.num_samples_per_class, num_iterations=args.num_iter, sam=True)
        else:
            trainloader, valloader, testloader = load_dataset(args.dataset, args.dataroot, 'pair',
                                                              batch_size=args.batch_size,
                                                              num_samples_per_class=args.num_samples_per_class, sam=True)
    else:
        if args.num_iter > 0:
            trainloader, valloader, testloader = load_dataset(args.dataset, args.dataroot, 'pair',
                                                              batch_size=args.batch_size,
                                                              num_samples_per_class=args.num_samples_per_class, num_iterations=args.num_iter)
        else:
            trainloader, valloader, testloader = load_dataset(args.dataset, args.dataroot, 'pair',
                                                              batch_size=args.batch_size,
                                                              num_samples_per_class=args.num_samples_per_class)



num_class = trainloader.dataset.num_classes
print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))
print('Number of test dataset: ' ,len(testloader.dataset))


# Model
print('==> Building model: {}'.format(args.model))

net = models.load_model(args.model, num_class)
# print(net)

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.csv')


# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

criterion = nn.CrossEntropyLoss()


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

kdloss = KDLoss(args.temp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_cls_loss = 0
    train_sam_loss = 0
    if not args.sam:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            batch_size = inputs.size(0)

            if not args.cls:
                outputs = net(inputs)
                loss = torch.mean(criterion(outputs, targets))
                train_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).sum().float().cpu()
            else:
                targets_ = targets[:batch_size//2]
                outputs = net(inputs[:batch_size//2])
                loss = torch.mean(criterion(outputs, targets_))
                train_loss += loss.item()

                with torch.no_grad():
                    outputs_cls = net(inputs[batch_size//2:])
                cls_loss = kdloss(outputs, outputs_cls.detach())
                loss += args.lamda * cls_loss
                train_cls_loss += cls_loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets_.size(0)
                correct += predicted.eq(targets_.data).sum().float().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | Cls: %.3f | Sam: %.3f'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total,
                            train_cls_loss/(batch_idx+1),
                            train_sam_loss/(batch_idx+1)))

    else:
        for batch_idx, (aug_dataset, raw_dataset) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = aug_dataset[0].cuda(), aug_dataset[1].cuda()
                inputs_sam = raw_dataset[0].cuda()

            batch_size = inputs.size(0)

            if not args.cls:
                outputs = net(inputs)
                loss = torch.mean(criterion(outputs, targets))
                train_loss += loss.item()

                with torch.no_grad():
                    outputs_sam = net(inputs_sam)
                sam_loss = kdloss(outputs, outputs_sam.detach())
                loss += args.lamda2 * sam_loss
                train_sam_loss += sam_loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().float()

            else:
                targets_ = targets[:batch_size//2]
                outputs = net(inputs[:batch_size//2])
                loss = torch.mean(criterion(outputs, targets_))
                train_loss += loss.item()

                with torch.no_grad():
                    outputs_cls = net(inputs[batch_size//2:])
                cls_loss = kdloss(outputs, outputs_cls.detach())
                loss += args.lamda * cls_loss
                train_cls_loss += cls_loss.item()

                with torch.no_grad():
                    outputs_sam = net(inputs_sam[:batch_size//2])
                sam_loss = kdloss(outputs, outputs_sam.detach())
                loss += args.lamda2 * sam_loss
                train_sam_loss += sam_loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets_.size(0)
                correct += predicted.eq(targets_.data).cpu().sum().float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | Cls: %.3f | Sam: %.3f'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total,
                            train_cls_loss/(batch_idx+1),
                            train_sam_loss/(batch_idx+1)))


    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [cls {:.3f}] [sam {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        train_loss/(batch_idx+1),
        train_cls_loss/(batch_idx+1),
        train_sam_loss/(batch_idx+1),
        100.*correct/total))

    return train_loss/batch_idx, 100.*correct/total, train_cls_loss/batch_idx, train_sam_loss/batch_idx

def val(epoch, test):
    global best_val
    global best_acc
    global best_acc2
    net.eval()
    val_loss = 0.0
    correct = 0.0
    correct2 = 0.0
    total = 0.0

    class_correct = torch.zeros(num_class)
    class_total = torch.zeros(num_class)

    # Define a data loader for evaluating
    if test:
        loader = testloader
    else:
        loader = valloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, requires_grad=False), Variable(targets, requires_grad=False)


            outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets))

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()

            for i in range(num_class):
                class_idx = targets.eq(i).float()
                class_total[i] += class_idx.cpu().sum()
                class_correct[i] += (predicted.eq(targets).float() * class_idx).cpu().sum()


            _, predict5 = torch.topk(outputs.data, k=5,dim=1,largest=True,sorted=True)
            for i in range(5):
                correct2 += predict5[:,i].eq(targets.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | Top5 Acc: %.3f%% (%d/%d) '
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*correct2/total, correct2, total))

    logger = logging.getLogger('test' if test else 'val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}] [Top 5 {:.3f}]'.format(
        epoch,
        val_loss/(batch_idx+1),
        100.*correct/total, 100.*correct2/total))


    acc = 100.*correct/total

    test_loss, test_acc = 0., 0.
    class_acc = 100. * class_correct / class_total

    if test:
        print("========== Class-wise test performance ( avg : {} ) ==========".format(torch.mean(class_acc)))
        print(class_acc)
    if acc > best_val and test == False:
        if args.dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
            best_val = acc
            checkpoint(acc, epoch)
            test_loss, test_acc, _, _, top5 = val(epoch, True)
            best_acc = test_acc
            best_acc2 = top5
        else:
            best_val = acc
            checkpoint(acc, epoch)
            test_loss = val_loss/(batch_idx+1)
            test_acc = acc
            best_acc = test_acc
            best_acc2 = 100.*correct2/total
            logger = logging.getLogger('test')
            logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}] [Top 5 {:.3f}]'.format(
                epoch,
                val_loss/(batch_idx+1),
                100.*correct/total, 100.*correct2/total))


    return (val_loss/(batch_idx+1), 100.*correct/total,
           test_loss, test_acc, 100.*correct2/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc',
                            'val loss', 'val acc',
                            'test loss', 'test acc',
                            'train cls loss',
                            'train sam loss'])

# Logs
for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc, train_cls_loss, train_sam_loss = train(epoch)
    val_loss, val_acc, test_loss, test_acc, class_acc = val(epoch, False)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, val_loss,
                            val_acc, test_loss, test_acc,
                            train_cls_loss, train_sam_loss])
    adjust_learning_rate(optimizer, epoch)

print("Best Accuracy : {}".format(best_acc))
logger = logging.getLogger('best')
logger.info('[Acc {:.3f}] [Top 5 {:.3f}]'.format(best_acc, best_acc2))
