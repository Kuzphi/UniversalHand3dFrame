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
from src.core.loss import CPMMSELoss as criterion
from src.utils.misc import MetricMeter, AverageMeter, to_torch, to_cuda, to_cpu, combine

def train(cfg, train_loader, model, optimizer, log):
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
        bar.suffix  = '({batch}/{size}) Data:{data:.1f}s Batch:{bt:.1f}s Total:{total:} ETA:{eta:} '\
                        'Loss:{loss:.4f} Acc:{acc: .3f} Dis:{dis:.3f}'.format(
                        batch=i + 1,
                        size=len(train_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc=acc.avg,
                        dis=dis.avg)
        bar.next()
    log.info(bar.suffix)
    bar.finish()
    return metric

def validate(cfg, val_loader, model, log = None):
    metric = MetricMeter(cfg.METRIC_ITEM)
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
            size = batch['weight'].size(0)

            # measure data loading time
            data_time.update(time.time() - end)
            
            # compute output

            outputs = model(to_cuda(batch['input']))


            #calculate loss
            if cfg.IS_INFERENCE:
                loss = criterion(outputs, batch)
                metric['loss'] = loss

            #convert output from cuda tensor to cpu tensor
            outputs = to_cpu(outputs)

            # debug, print intermediate result
            if cfg.DEBUG:
                debug(outputs, batch, loss)

            if cfg.IS_INFERENCE:
                metric_ = val_loader.dataset.eval_result(outputs, batch, cfg = cfg)
                metric_['loss'] = loss.item()
                metric.update(metric_, size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            preds = val_loader.dataset.get_preds(outputs)
            all_preds.append(preds)

            bar.suffix  = '({batch}/{size}) Data:{data:.1f}s Batch:{bt:.1f}s Total:{total:} ETA:{eta:}' \
                            'Loss:{loss:.4f} Acc:{acc: .4f} Dis:{dis:.3f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.val,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            acc=acc.avg,
                            dis=dis.avg)
            bar.next()
        log.info(bar.suffix)
        bar.finish()
        
    if cfg.IS_INFERENCE:
        metric = {  'loss':losses.avg, 
                    'acc':acc.avg,
                    'dis': dis.avg}
        return metric, reduce(combine, all_preds)

    return reduce(combine, all_preds)

def inference(cfg, infer_loader, model):
    data_time = AverageMeter()    
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(infer_loader.dataset)
    all_preds = []

    idx = 0
    bar = Bar('Processing', max=len(infer_loader))
    with torch.no_grad():
        end = time.time()
        idx = 0
        for i, batch in enumerate(infer_loader):
            data_time.update(time.time() - end)

            # compute output
            outputs = model(to_cuda(batch['input']))
            outputs = to_cpu(outputs)

            # debug(outputs, batch, 1)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # preds = get_preds(output)
            preds = infer_loader.dataset.get_preds(outputs)
            all_preds.append(preds)

            bar.suffix  = '({batch}/{size}) Data:{data:.1f}s Batch:{bt:.1f}s Total:{total:} ETA:{eta:}'.format(
                            batch=i + 1,
                            size=len(infer_loader),
                            data=data_time.val,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td)
            bar.next()
        bar.finish()
    return reduce(combine, all_preds)
