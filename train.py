# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(Kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import torch
from torch.utils.data import DataLoader

import src.core.loss as  loss
from src import model
from src import dataset
from src.core import train, validate, debug
from src.core.log import Log
from src.utils.misc import get_config, save_checkpoint
def main(args):
    print("Reading configuration file")
    cfg = get_config(args.cfg)
    cfg.DEBUG = args.debug

    print("Loading Training Data")
    train_data = eval('dataset.' + cfg.TRAIN.DATASET.NAME)(cfg.TRAIN.DATASET)
    valid_data = eval('dataset.' + cfg.VALID.DATASET.NAME)(cfg.VALID.DATASET)

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.TRAIN.DATASET.BATCH_SIZE * len(cfg.GPUS),
        shuffle=cfg.TRAIN.DATASET.SHUFFLE,
        num_workers=cfg.WORKERS)

    valid_loader = DataLoader(
        valid_data,
        batch_size=cfg.VALID.DATASET.BATCH_SIZE * len(cfg.GPUS),
        num_workers=cfg.WORKERS)
    
    print("Loding Loss")
    criterion = eval('loss.' + cfg.CRITERION)

    print("Creating Model")
    model = eval('model.' + cfg.MODEL.NAME)(**cfg.MODEL)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    if cfg.MODEL.PRETRAINED_WEIGHT_PATH:
        print("Loading Pretrained Weight")
        weight = torch.load(cfg.MODEL.PRETRAINED_WEIGHT_PATH)
        model.load_state_dict(weight)

    print("Creating optimizer and optimizer scheduler")
    optimizer = eval('torch.optim.' + cfg.OPTIMIZER.NAME)(model.parameters(),**cfg.OPTIMIZER.PARAMETERS)
    scheduler = eval('torch.optim.lr_scheduler.' + cfg.OPTIMIZER_SCHEDULE.NAME)(optimizer, **cfg.OPTIMIZER_SCHEDULE.PARAMETERS)

    print("Creating log")
    log = Log(cfg.LOG.PATH, monitor_item = cfg.LOG.MONITOR_ITEM, title = cfg.TAG)

    if cfg.RESUME_TRAIN:
        print("Resuming data from checkpoint")
        checkpoint = torch.load(cfg.CHEKCPOINT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    best = 1e99
    sgn = -1 if cfg.MAIN_METRIC.endswith('acc') else 1

    for epoch in range(cfg.START_EPOCH, cfg.END_EPOCH):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print('\nEpoch: %d | LR:' % (epoch), lr)

        # train for one epoch
        train_metric = train(cfg.TRAIN, train_loader, model, criterion, optimizer, log)

        # evaluate on validation set
        valid_metric, predictions = validate(cfg.VALID,valid_loader, model, criterion, log)

        # append logger file value should be basic type for json serialized
        epoch_result = {}
        for item in cfg.LOG.MONITOR_ITEM:
            if item == 'lr':
                epoch_result['lr'] = float(lr)
                continue
            x, y = item.split('_')
            if x == 'train':
                epoch_result[item] = train_metric[y].avg
            if x == 'valid':
                epoch_result[item] = valid_metric[y].avg

        log.append(epoch_result)

        # remember best acc and save checkpoint
        new_metric = valid_metric[cfg.MAIN_METRIC].avg
        is_best = sgn * new_metric < best  #if loss then < if acc then > !!!
        best = min(best, sgn * new_metric)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best': sgn * best,
            'optimizer' : optimizer.state_dict(),
        }, predictions, cfg, log, is_best, fpath=cfg.CHECKPOINT, snapshot = 30)

        cfg.CURRENT_EPOCH = epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='configure file',
                        type=str)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')

    args = parser.parse_args()
    main(args)