# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(Kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
import src.core.loss as loss

from src import model
from src import dataset
from src.core import validate
from src.utils.misc import get_config, save_infer_result

def main(args):
    print("Reading configuration file")
    cfg = get_config(args.cfg, type = 'infer')

    print("Loading Training Data")
    infer_data = eval('dataset.' + cfg.DATASET.NAME)(cfg.DATASET)

    infer_loader = DataLoader(
        infer_data,
        batch_size=cfg.DATASET.BATCH_SIZE * len(cfg.GPUS),
        shuffle=cfg.DATASET.SHUFFLE,
        num_workers=cfg.WORKERS)

    print("Loding Loss")
    criterion = eval('loss.' + cfg.CRITERION)

    print("Creating Model")
    model = eval('model.' + cfg.MODEL.NAME)(**cfg.MODEL)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda().eval()

    print("Loading Pretrained Weight")
    weight = torch.load(cfg.MODEL.PRETRAINED_WEIGHT_PATH)
    if weight.has_key('state_dict'):
        weight = weight['state_dict']
    model.load_state_dict(weight)

    print("Starting Inference")
    if cfg.IS_VALID:
        metric, preds = validate(cfg, infer_loader, model, criterion)
        save_infer_result(preds, metric, cfg.CHECKPOINT)
    else:
        preds = validate(cfg, infer_loader, model, criterion)
        save_infer_result(preds, None, cfg.CHECKPOINT)

    if cfg.POST_INFER:
        infer_data.post_infer(cfg, preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='configure file',
                        type=str)
    args = parser.parse_args()
    main(args)