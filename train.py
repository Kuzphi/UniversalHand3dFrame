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
from src.utils.misc import MetricMeter, get_config, save_checkpoint
def main(args):
    print("Reading configuration file")
    cfg = get_config(args.cfg)
    cfg.DEBUG = args.debug

    print("Loading Training Data")
    train_data = eval('dataset.' + cfg.TRAIN.DATASET.NAME)(cfg.TRAIN.DATASET)
    print("Loading Valid Data")
    valid_data = eval('dataset.' + cfg.VALID.DATASET.NAME)(cfg.VALID.DATASET)
    print ("Train Data Size: ", len(train_data))
    print ("Valid Data Size: ", len(valid_data))
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.TRAIN.DATASET.BATCH_SIZE * len(cfg.MODEL.GPUS),
        shuffle=cfg.TRAIN.DATASET.SHUFFLE,
        num_workers=cfg.WORKERS)

    valid_loader = DataLoader(
        valid_data,
        batch_size=cfg.VALID.DATASET.BATCH_SIZE * len(cfg.MODEL.GPUS),
        num_workers=cfg.WORKERS)

    print("Creating Model")
    model = eval('model.' + cfg.MODEL.NAME)(cfg.MODEL)

    print("Creating Log")
    log = Log(cfg.LOG.PATH, monitor_item = cfg.LOG.MONITOR_ITEM, metric_item = cfg.METRIC_ITEMS, title = cfg.TAG)

    if cfg.RESUME_TRAIN:
        print("Resuming data from checkpoint")
        checkpoint = torch.load(cfg.CHEKCPOINT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    best = 1e99
    sgn = -1 if cfg.MAIN_METRIC.endswith('Acc') else 1

    for epoch in range(cfg.START_EPOCH, cfg.END_EPOCH):
        epoch_result = {}

        print('\nEpoch: %d: | LR' % (epoch), end = ' ')

        for name, scheduler in model.schedulers.iteritems():
            print(name, ':', scheduler.get_lr()[0], end = ' ')
            epoch_result[name + '_lr'] = scheduler.get_lr()[0]
        print('')

        # train for one epoch
        train_metric = MetricMeter(cfg.METRIC_ITEMS)
        train(cfg.TRAIN, train_loader, model, train_metric, log)

        # evaluate on validation set
        valid_metric = MetricMeter(cfg.METRIC_ITEMS)
        predictions = validate(cfg.VALID,valid_loader, model, valid_metric, log)

        # append logger file value should be basic type for json serialized
        
        for item in cfg.LOG.MONITOR_ITEM:
            x , y = item.split('_')
            if x == 'train':
                epoch_result[item] = train_metric[y].avg
            if x == 'valid':
                epoch_result[item] = valid_metric[y].avg

        log.append(epoch_result)

        # remember best acc and save checkpoint
        new_metric = valid_metric[cfg.MAIN_METRIC].avg
        is_best = sgn * new_metric < best
        best = min(best, sgn * new_metric)
        cfg.CURRENT_EPOCH = epoch

        save_checkpoint(model, predictions, cfg, log, is_best, fpath=cfg.CHECKPOINT, snapshot = 5)

        
        model.update_learning_rate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='configure file',
                        type=str)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')

    args = parser.parse_args()
    main(args)