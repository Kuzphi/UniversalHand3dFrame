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
from src.core.evaluate import eval_result, get_preds
from src.core.loss import CPMMSELoss as criterion
from src.utils.misc import AverageMeter, to_torch

def train(cfg, train_loader, model, optimizer, log):
    acc = AverageMeter()
    dis = AverageMeter()
    losses = AverageMeter()
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
        outputs = model(batch['input'])

        #calculate loss
        loss = criterion(outputs, batch)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #convert output from cuda tensor to cpu tensor
        outputs = [out.detach().cpu() for out in outputs]

        # debug, print intermediate result
        if cfg.DEBUG:
            debug(outputs[-1], batch, loss)

        # measure accuracy and record loss
        losses.update(loss.item(), size)

        avg_acc, avg_dis = eval_result(outputs[-1], batch, num_joints = cfg.DATASET.NUM_JOINTS)
        acc.update(avg_acc[0], size)
        dis.update(avg_dis[0], size)

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
    metric = {  'loss':losses.avg, 
                'acc':acc.avg,
                'dis': dis.avg}
    return metric

def validate(cfg, val_loader, model, log):
    dis = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()    
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_loader.dataset)
    all_preds = np.zeros((num_samples, cfg.DATASET.NUM_JOINTS, cfg.DATASET.DIM_JOINT),
                         dtype=np.float32)

    idx = 0
    bar = Bar('Processing', max=len(val_loader))
    with torch.no_grad():
        idx = 0
        end = time.time()
        for i, batch in enumerate(val_loader):
            size = batch['weight'].size(0)

            # measure data loading time
            data_time.update(time.time() - end)
            
            # compute output
            outputs = model(batch['input'])

            #calculate loss
            loss = criterion(outputs, batch)

            #convert output from cuda tensor to cpu tensor
            outputs = [out.detach().cpu() for out in outputs]

            # debug, print intermediate result
            if cfg.DEBUG:
                debug(outputs[-1], batch, loss)

            # measure accuracy and record loss
            losses.update(loss.item(), size)

            avg_acc, avg_dis = eval_result(outputs[-1], batch, num_joints = cfg.DATASET.NUM_JOINTS)
            acc.update(avg_acc[0], size)
            dis.update(avg_dis[0], size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            preds = get_preds(outputs[-1])
            all_preds[idx:idx + size, :, :] = preds

            idx += size

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
    metric = {  'loss':losses.avg, 
                'acc':acc.avg,
                'dis': dis.avg}
    return metric, all_preds

def inference(cfg, infer_loader, model):
    dis = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(infer_loader.dataset)
    all_preds = np.zeros((num_samples, cfg.DATASET.NUM_JOINTS, 3),
                         dtype=np.float32)

    idx = 0
    bar = Bar('Processing', max=len(infer_loader))
    with torch.no_grad():
        end = time.time()
        idx = 0
        for i, batch in enumerate(infer_loader):
            # compute output
            output = model(batch['input'])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            preds = get_preds(output)

            size = preds.size(0)

            all_preds[idx:idx + size, :, :] = preds
            idx += input.size(0)

            bar.suffix  = '({batch}/{size}) Data:{data:.1f}s Batch:{bt:.1f}s Total:{total:} ETA:{eta:} \
                            Loss:{loss:.4f} Acc:{acc: .4f} Dis:{dis:.3f}'.format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.val,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            acc=acces.avg,
                            dis=dis.avg)
            bar.next()
        log.info(bar.suffix)
        bar.finish()
    metric = edict({'acc':acc.avg,'dis': dis.avg})
    return metric, all_preds
