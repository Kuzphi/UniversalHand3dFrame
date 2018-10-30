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
import src.core.loss as  loss

from src import model
from src import dataset
from src.core import validate
from src.utils.misc import get_config, save_preds

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
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    print("Loading Pretrained Weight")
    weight = torch.load(cfg.MODEL.PRETRAINED_WEIGHT_PATH)
    model.load_state_dict(weight)

    print("Starting Inference")
    preds = validate(cfg, infer_loader, model, criterion)
    save_preds(preds, cfg.CHECKPOINT)

    if cfg.DARW_RESULT:
        fpath = os.path.join(cfg.CHECKPOINT, 'image')
        os.makedirs(fpath)
        infer_data.preds_demo(preds, fpath)


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='configure file',
                        type=str)
    args = parser.parse_args()
    main(args)