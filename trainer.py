'''
Copyright (c) 2018, Yerlan Idelbayev

Redistribution and use in source and binary forms, with or without modification, are 
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of 
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided 
with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE.
'''

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from quant import convert, quant_param_groups, QuantSAM, OscProbe

# Factories are named "<dataset>_<arch>" (e.g. cifar100_resnet20); the arch set
# is identical for both datasets, so derive it from the cifar100_ prefix.
model_names = sorted(name[len('cifar100_'):] for name in dir(models)
                     if name.startswith('cifar100_'))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10/100 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                    choices=['cifar10', 'cifar100'],
                    help='cifar10 | cifar100 (default: cifar100)')
parser.add_argument('--data', metavar='DIR',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                    help='dataset root, holding cifar-{10-batches,100}-python '
                         '(default: <repo>/data)')
parser.add_argument('--download', dest='download', action='store_true',
                    help='download the dataset into --data if it is missing')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--init-from', default='', type=str, metavar='PATH',
                    help='load weights from a .th/.pt file before training '
                         '(use to start QAT from your own FP32 baseline)')
parser.add_argument('--seed', default=0, type=int, metavar='N',
                    help='random seed (default: 0)')

# --- quantization-aware training (LSQ) ---
parser.add_argument('--qat', action='store_true',
                    help='enable LSQ quantization-aware training')
parser.add_argument('--w-bits', default=4, type=int, metavar='B',
                    help='weight bits, 0 keeps weights FP32 (default: 4)')
parser.add_argument('--a-bits', default=4, type=int, metavar='B',
                    help='activation bits, 0 keeps activations FP32 (default: 4)')
parser.add_argument('--q-first-last-bits', default=8, type=int, metavar='B',
                    help='bits for the stem conv and the classifier, 0 leaves '
                         'them FP32 (default: 8, the LSQ/PACT/DoReFa convention)')
parser.add_argument('--no-per-channel', dest='q_per_channel', action='store_false',
                    help='per-tensor weight quantization (default: per-channel)')
parser.add_argument('--a-signed', default='auto',
                    choices=['auto', 'always'],
                    help='activation signedness policy (default: auto)')
parser.add_argument('--q-init', default='lsq', choices=['lsq', 'minmax'],
                    help='step-size initialization: lsq = 2<|v|>/sqrt(Qp) '
                         '(paper, tuned for low bits), minmax = max/Qp '
                         '(default: lsq)')
parser.add_argument('--q-step-lr-scale', default=1.0, type=float, metavar='F',
                    help='LR multiplier for the LSQ step sizes (default: 1.0)')
parser.add_argument('--sam', action='store_true',
                    help='QuantSAM: ascent by flipping rounding decisions')
parser.add_argument('--sam-budget', default='cost', choices=['cost', 'gain'],
                    help='what --sam-rho caps: cost = ||eps||_2 radius, '
                         'gain = first-order loss increase (default: cost)')
parser.add_argument('--sam-rho', default=0.05, type=float, metavar='R',
                    help='SAM budget, read per --sam-budget (default: 0.05)')
parser.add_argument('--osc-probe', action='store_true',
                    help='log oscillation frequency and amplitude')
parser.add_argument('--osc-momentum', default=0.01, type=float, metavar='M',
                    help='EMA momentum for the oscillation probe (default: 0.01)')
parser.add_argument('--osc-out', default='', type=str, metavar='PATH',
                    help='write probe history/traces to PATH.npz for plot_osc.py')
parser.add_argument('--osc-label', default='', type=str,
                    help='run label used in the figures')
parser.add_argument('--osc-fig', default='', type=str, metavar='DIR',
                    help='also render this run figures into DIR at the end')
parser.add_argument('--osc-trace-epochs', default=5, type=int, metavar='N',
                    help='trace the rounded levels over the last N epochs (default: 5)')
parser.add_argument('--osc-trace-layer', default=-1, type=int, metavar='I',
                    help='quantized layer to trace; -1 picks the middle one')
parser.add_argument('--osc-top-k', default=5, type=int, metavar='K',
                    help='oscillating weights to draw in the trace figure (default: 5)')

# cifar100 values are taken verbatim from the recipe that produced the
# pytorch-cifar-models pretrained weights (conf/cifar100.conf).
DATASET_STATS = {
    'cifar10': dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'cifar100': dict(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
}


def check_writable(path):
    """Fail before training rather than after it."""
    d = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(d, exist_ok=True)
    stamp = os.path.join(d, '.write_check')
    open(stamp, 'w').close()
    os.remove(stamp)


def load_weights(model, path):
    obj = torch.load(path, map_location='cpu')
    state = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
    if len(state) > 0 and all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    print("=> loaded weights from '{}'".format(path))


def main():
    global args
    args = parser.parse_args()
    print('=> {} | {} | available archs: {}'.format(
        args.dataset, args.arch, ' '.join(model_names)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = getattr(models, f'{args.dataset}_{args.arch}')(pretrained=args.pretrained)
    if args.init_from:
        load_weights(model, args.init_from)

    model.cuda()
    cudnn.benchmark = True

    dataset_cls = datasets.CIFAR100 if args.dataset == 'cifar100' else datasets.CIFAR10
    normalize = transforms.Normalize(**DATASET_STATS[args.dataset])

    train_loader = torch.utils.data.DataLoader(
        dataset_cls(root=args.data, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=args.download),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset_cls(root=args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=args.download),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.qat:
        convert(model, w_bits=args.w_bits, a_bits=args.a_bits,
                first_last_bits=args.q_first_last_bits,
                per_channel=args.q_per_channel,
                a_signed=args.a_signed,
                init_mode=args.q_init)
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    params = quant_param_groups(model, args) if args.qat else model.parameters()
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    sam = QuantSAM(model, rho=args.sam_rho,
                   budget=args.sam_budget) if args.sam else None
    probe = OscProbe(
        model, momentum=args.osc_momentum,
        trace_steps=args.osc_trace_epochs * len(train_loader),
        trace_layer=None if args.osc_trace_layer < 0 else args.osc_trace_layer,
    ) if args.osc_probe else None

    if probe is not None:
        for p in filter(None, (args.osc_out, args.osc_fig and
                               os.path.join(args.osc_fig, 'x'))):
            check_writable(p)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    best_prec1, best_prec5 = 0, 0
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, sam, probe)
        lr_scheduler.step()
        if probe is not None:
            probe.snapshot()

        # evaluate on validation set; top-5 is reported at the best top-1
        # epoch, so that both numbers describe the same model
        prec1, prec5 = validate(val_loader, model, criterion)
        if prec1 > best_prec1:
            best_prec1, best_prec5 = prec1, prec5
        print(' * best so far  Prec@1 {:.3f}  Prec@5 {:.3f}'
              .format(best_prec1, best_prec5))

    if probe is not None and args.osc_out:
        label = args.osc_label or os.path.splitext(os.path.basename(args.osc_out))[0]
        out = args.osc_out
        try:
            probe.save(out, label=label)
        except OSError as e:
            out = os.path.basename(args.osc_out)
            print('!! {} failed ({}), falling back to {}'.format(args.osc_out, e, out))
            probe.save(out, label=label)
        print('=> wrote oscillation probe to {}'.format(out))
        if args.osc_fig:
            try:
                from plot_osc import make_figures
                run = (label, np.load(out, allow_pickle=True))
                for p in make_figures([run], args.osc_fig, prefix=label + '_',
                                      n_show=args.osc_top_k):
                    print('=> {}'.format(p))
            except Exception as e:
                print('!! figures failed ({}); replot from {}'.format(e, out))

    print(' * final  Prec@1 {:.3f}  Prec@5 {:.3f}'.format(prec1, prec5))
    print(' * best   Prec@1 {:.3f}  Prec@5 {:.3f}'.format(best_prec1, best_prec5))


def train(train_loader, model, criterion, optimizer, epoch, sam=None, probe=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        osc = probe.step() if probe is not None else None
        if sam is not None:
            # meters keep the clean-weight loss from the first pass
            sam.ascent_step()
            optimizer.zero_grad()
            criterion(model(input_var), target_var).backward()
            sam.restore()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
            if osc is not None:
                print('    ' + OscProbe.format(osc))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # .reshape, not .view: correct[:k] is non-contiguous for k > 1
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()