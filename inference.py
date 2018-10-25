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

from src import model
from src import dataset
from src.core import inference
from src.utils.misc import get_config, save_preds
from src.utils.imutils import 
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
    
    print("Creating Model")
    model = eval('model.' + cfg.MODEL.NAME)(**cfg.MODEL)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    print("Loading Pretrained Weight")    
    weight = torch.load(cfg.MODEL.PRETRAINED_WEIGHT_PATH)
    model.load_state_dict(weight)

    print("Starting Inference")
    preds = inference(cfg, infer_loader, model)
    save_preds(preds, cfg.CHECKPOINT)

    if cfg.SAVE_IMG_RESULT:
        os.path.mkdirs(cfg.)
    if cfg.DARW_RESULT:
        # to-do draw all the result here
        for batch, pred in zip(infer_data, preds)
            result = infer_data.demo_result(batch, preds)
            if cfg.SAVE_IMG_RESULT:
                cv2.imwrite(result, cfg.)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='configure file',
                        type=str)
    args = parser.parse_args()
    main(args)