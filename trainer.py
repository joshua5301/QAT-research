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
from quant import (convert, quant_param_groups, DiscreteSAM, SAQ, OOQ, AOQ,
                   FlipProbe)

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
parser.add_argument('--ewgs', default=0.0, type=float, metavar='D',
                    help='EWGS gradient scaling delta, in grid units where the '
                         'rounding error spans +-0.5; 0 keeps the plain STE, '
                         '2 is the largest value that keeps the scale '
                         'non-negative (default: 0.0)')
parser.add_argument('--dsam', action='store_true',
                    help='DiscreteSAM: ascent by flipping rounding decisions')
parser.add_argument('--dsam-rho', default=0.05, type=float, metavar='R',
                    help='first-order loss increase the ascent buys, in nats; '
                         'independent of step size and bit width (default: 0.05)')
parser.add_argument('--dsam-cont', default='bn,bias', type=str, metavar='LIST',
                    help='continuous parameter groups sharing the DiscreteSAM '
                         'budget, comma separated from bn,bias,wscale,ascale; '
                         'empty flips rounding decisions only (default: bn,bias, '
                         'matching --saq-cont)')
parser.add_argument('--saq', action='store_true',
                    help='SAQ: SAM with the perturbation on the quantized weights')
parser.add_argument('--saq-rho', default=0.05, type=float, metavar='R',
                    help='l2 radius of the SAQ ball. The reference uses 0.4, but on '
                         'weight-normalized weights, so it does not carry over '
                         'directly to raw LSQ (default: 0.05)')
parser.add_argument('--saq-cont', default='bn,bias', type=str, metavar='LIST',
                    help='continuous parameter groups inside the SAQ ball, comma '
                         'separated from bn,bias,wscale,ascale; empty perturbs the '
                         'quantized weights only. The default matches the reference '
                         '(include_bn=True, both clips off)')
parser.add_argument('--ooq', action='store_true',
                    help='OOQ: oscillation dampening, a quadratic pull onto the grid')
parser.add_argument('--ooq-lambda', default=0.1, type=float, metavar='L',
                    help='dampening weight at the end of training, on the '
                         'published scale (the paper anneals 0 -> 0.1) (default: 0.1)')
parser.add_argument('--ooq-lambda-start', default=0.0, type=float, metavar='L',
                    help='dampening weight before the ramp starts (default: 0.0)')
parser.add_argument('--ooq-anneal-start', default=0.25, type=float, metavar='F',
                    help='fraction of training held at --ooq-lambda-start before '
                         'the cosine ramp begins (default: 0.25, as released)')
parser.add_argument('--aoq', action='store_true',
                    help='AOQ: contract the grid to explore, release, then dampen')
parser.add_argument('--aoq-stage', default='0.2,0.6', type=str, metavar='F1,F2',
                    help='stage boundaries as fractions of --epochs; the reference '
                         'uses epochs 50 and 150 of 250 (default: 0.2,0.6)')
parser.add_argument('--aoq-alpha-min', default=0.3, type=float, metavar='A',
                    help='grid contraction at the end of the explore stage '
                         '(default: 0.3, as the reference cosine bottoms out)')
parser.add_argument('--aoq-lambda', default=0.01, type=float, metavar='L',
                    help='dampening weight during the last stage (default: 0.01)')
parser.add_argument('--flip-probe', action='store_true',
                    help='report per layer where the DiscreteSAM flips land')
parser.add_argument('--flip-out', default='', type=str, metavar='PATH',
                    help='write per-epoch per-layer flip counts to PATH.npz')

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
                init_mode=args.q_init,
                ewgs=args.ewgs)
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    params = quant_param_groups(model, args) if args.qat else model.parameters()
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    assert not (args.dsam and args.saq), 'pick one of --dsam / --saq'
    sam = (DiscreteSAM(model, rho=args.dsam_rho,
                       cont=filter(None, args.dsam_cont.split(','))) if args.dsam else
           SAQ(model, rho=args.saq_rho,
               cont=filter(None, args.saq_cont.split(','))) if args.saq else None)
    assert not (args.ooq and args.aoq), 'pick one of --ooq / --aoq'
    s1, s2 = (float(v) for v in args.aoq_stage.split(','))
    damp = (OOQ(model, args.ooq_lambda, args.epochs,
                lam_start=args.ooq_lambda_start,
                anneal_start=args.ooq_anneal_start) if args.ooq else
            AOQ(model, args.epochs, stage1=s1, stage2=s2,
                alpha_min=args.aoq_alpha_min,
                damp_lam=args.aoq_lambda) if args.aoq else None)
    probe = FlipProbe(model) if args.flip_probe else None

    assert probe is None or args.dsam, '--flip-probe needs --dsam'
    if probe is not None and args.flip_out:
        check_writable(args.flip_out)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    best_prec1, best_prec5 = 0, 0
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        if args.aoq:
            damp.set_epoch(epoch)
            print('aoq: stage {}  alpha {:.4f}'
                  .format(damp.stage(epoch), damp.alpha(epoch)))
        train(train_loader, model, criterion, optimizer, epoch, sam, probe, damp)
        lr_scheduler.step()
        if probe is not None:
            probe.snapshot()
            print(probe.report())

        # evaluate on validation set; top-5 is reported at the best top-1
        # epoch, so that both numbers describe the same model
        prec1, prec5 = validate(val_loader, model, criterion)
        if prec1 > best_prec1:
            best_prec1, best_prec5 = prec1, prec5
        print(' * best so far  Prec@1 {:.3f}  Prec@5 {:.3f}'
              .format(best_prec1, best_prec5))

    if probe is not None and args.flip_out:
        probe.save(args.flip_out,
                   label=os.path.splitext(os.path.basename(args.flip_out))[0])
        print('=> wrote flip counts to {}'.format(args.flip_out))

    print(' * final  Prec@1 {:.3f}  Prec@5 {:.3f}'.format(prec1, prec5))
    print(' * best   Prec@1 {:.3f}  Prec@5 {:.3f}'.format(best_prec1, best_prec5))


def train(train_loader, model, criterion, optimizer, epoch, sam=None, probe=None,
          damp=None):
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
        if sam is not None:
            # meters keep the clean-weight loss from the first pass
            sam.ascent_step()
            if probe is not None:
                probe.step()
            optimizer.zero_grad()
            criterion(model(input_var), target_var).backward()
            sam.restore()
        if damp is not None:
            p = damp.penalty(epoch)
            if p is not None:
                p.backward()
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
            if damp is not None:
                print('    damp: lambda {:.4g}  penalty {:.5g}'
                      .format(damp.weight(epoch), float(damp.last)))


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