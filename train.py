import argparse
from functools import partial
from importlib import import_module
import json
import os
import random
import re
import shutil
import time

import numpy as np
import cupy as cp

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.datasets import TransformDataset
from chainer.datasets import cifar
import chainer.links as L
from chainer.training import extensions
from chainercv import transforms

from utilities import ConvRegularization

USE_OPENCV = False


def transform(inputs, mean, std, crop_size=(32, 32), train=True):
    img, label = inputs
    img = img.copy()

    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]

    if train:
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # zero_pad
        img = np.pad(img, ((0,), (4,), (4,)), 'constant')
        # Random crop
        if tuple(crop_size) != (40, 40):
            img = transforms.random_crop(img, tuple(crop_size))

    return img, label


def create_result_dir(prefix):
    result_dir = 'results/{}_{}_0'.format(
        prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    shutil.copy(__file__, os.path.join(result_dir, os.path.basename(__file__)))
    return result_dir


def run_training(
        net, train, valid, result_dir, batchsize=64, devices=-1,
        training_epoch=300, initial_lr=0.05, lr_decay_rate=0.5,
        lr_decay_epoch=30, weight_decay=0.0005, regul_freq=0.5):
    # Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    test_iter = iterators.MultiprocessIterator(valid, batchsize, False, False)

    # Model
    net = L.Classifier(net)

    # Optimizer
    optimizer = optimizers.MomentumSGD(lr=initial_lr)
    optimizer.setup(net)

    optimizer.add_hook(ConvRegularization(regul_freq, devices), timing='post')

    if weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    # Updater
    if isinstance(devices, int):
        devices['main'] = devices
        updater = chainer.training.updater.StandardUpdater(
            train_iter, optimizer, device=devices)
    elif isinstance(devices, dict):
        updater = chainer.training.updater.ParallelUpdater(
            train_iter, optimizer, devices=devices)

    # 6. Trainer
    trainer = training.Trainer(
        updater, (training_epoch, 'epoch'), out=result_dir)

    # 7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(
        test_iter, net, device=devices['main']), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss',
         'val/main/accuracy', 'elapsed_time', 'lr']))
    lr_decay_trigger = training.triggers.ManualScheduleTrigger(lr_decay_epoch, 'epoch')
    trainer.extend(extensions.ExponentialShift(
        'lr', lr_decay_rate), trigger=lr_decay_trigger)
    trainer.extend(extensions.snapshot(), trigger=(50, 'epoch'))
    trainer.run()

    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='models/procresnet.py')
    parser.add_argument('--model_name', type=str, default='ProcResNet166')
    parser.add_argument('--gpus', type=int, nargs='*', default=[0, 1])
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--regul_freq', type=float, default=0.5)

    # Train settings
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--training_epoch', type=int, default=300)
    parser.add_argument('--initial_lr', type=float, default=0.1)
    parser.add_argument('--lr_decay_epoch', type=float, nargs='*', default=[150, 225])
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    # Data augmentation settings
    parser.add_argument('--crop_size', type=int, nargs='*', default=[32, 32])

    args = parser.parse_args()

    # Set the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    if len(args.gpus) > 1 or args.gpus[0] >= 0:
        chainer.cuda.cupy.random.seed(args.seed)
        gpus = {'main': args.gpus[0]}
        if len(args.gpus) > 1:
            gpus.update({'gpu{}'.format(i): i for i in args.gpus[1:]})
        args.gpus = gpus

    # print gpus
    with cp.cuda.Device(gpus['main']):

        if args.dataset == 'cifar10':
            train, valid = cifar.get_cifar10(scale=255.)
            n_class = 10
        elif args.dataset == 'cifar100':
            train, valid = cifar.get_cifar100(scale=255.)
            n_class = 100

        # Enable autotuner of cuDNN
        chainer.config.autotune = True
        # Load model
        ext = os.path.splitext(args.model_file)[1]
        mod_path = '.'.join(os.path.split(args.model_file.replace(ext, '')))
        mod = import_module(mod_path)
        net = getattr(mod, args.model_name)(n_class=n_class)

        # create result dir
        result_dir = create_result_dir(args.model_name)
        shutil.copy(args.model_file, os.path.join(
            result_dir, os.path.basename(args.model_file)))
        with open(os.path.join(result_dir, 'args'), 'w') as fp:
            fp.write(json.dumps(vars(args)))
        print(json.dumps(vars(args), sort_keys=True, indent=4))

        mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
        std = np.std([x for x, _ in train], axis=(0, 2, 3))

        train_transform = partial(
            transform, mean=mean, std=std, crop_size=args.crop_size, train=True)
        valid_transform = partial(transform, mean=mean, std=std, train=False)

        train = TransformDataset(train, train_transform)
        valid = TransformDataset(valid, valid_transform)

        run_training(
            net, train, valid, result_dir, args.batchsize, args.gpus,
            args.training_epoch, args.initial_lr, args.lr_decay_rate,
            args.lr_decay_epoch, args.weight_decay, args.regul_freq)