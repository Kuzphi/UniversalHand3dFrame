# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

from easydict import EasyDict as edict
import numpy as np
import torch

from src import Bar
from src.core.debug import debug
from src.core.evaluate import eval_result
from src.utils.misc import MetricMeter, AverageMeter, to_torch, to_cuda, to_cpu, combine

def train(cfg, train_loader, model, criterion, optimizer, log):
    metric = MetricMeter(cfg.METRIC_ITEM)
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    for i, batch in enumerate(train_loader):
        size = batch['weight'].size(0)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(to_cuda(batch['input']))

        #calculate loss
        loss = criterion(outputs, batch)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #convert output from cuda tensor to cpu tensor
        outputs = to_cpu(outputs) #[out.detach().cpu() for out in outputs]

        # debug, print intermediate result
        if cfg.DEBUG:
            debug(outputs, batch, loss)

        # measure accuracy and record loss
        metric_ = train_loader.dataset.eval_result(outputs, batch, cfg = cfg)
        metric_['loss'] = loss.item() 
        metric.update(metric_)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        suffix = '({batch}/{size}) Data:{data:.1f}s Batch:{bt:.1f}s Total:{total:} ETA:{eta:}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.val,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td)
        for name in metric_:
            suffix += '{}: {}'.format(name, metric_[name])
        bar.next()
    log.info(bar.suffix)
    bar.finish()
    return metric

def validate(cfg, val_loader, model, criterion, log = None):
    metric = MetricMeter(cfg.METRIC_ITEMS)
    data_time = AverageMeter()    
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_loader.dataset)
    all_preds = []

    idx = 0
    bar = Bar('Processing', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):

            data_time.update(time.time() - end)

            size = batch['weight'].size(0)

            # measure data loading time
            data_time.update(time.time() - end)
            
            # compute output

            outputs = model(to_cuda(batch['input']))


            #calculate loss
            if cfg.IS_VALID:
                loss = criterion(outputs, batch)

            #convert output from cuda tensor to cpu tensor
            outputs = to_cpu(outputs)

            # debug, print intermediate result
            if cfg.DEBUG:
                debug(outputs, batch, loss)

            if cfg.IS_VALID:
                metric_ = val_loader.dataset.eval_result(outputs, batch)
                metric_['loss'] = loss.item()
                metric.update(metric_, size)

            preds = val_loader.dataset.get_preds(outputs)
            all_preds.append(preds)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            suffix = '({batch}/{size}) Data:{data:.1f}s Batch:{bt:.1f}s Total:{total:} ETA:{eta:}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td)
            for name in metric.metric.keys():
                suffix += ' {}: {:.4f}'.format(name, metric.metric[name].avg)

            bar.suffix  = suffix
            bar.next()

        if log is not None:
            log.info(bar.suffix)
        bar.finish()

    if cfg.IS_VALID:
        return metric, reduce(combine, all_preds)

    return reduce(combine, all_preds)