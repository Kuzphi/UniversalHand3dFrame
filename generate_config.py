# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml
import time
from easydict import EasyDict as edict

def config_sample():
	config = dict()
	config['START_EPOCH'] = 0
	config['END_EPOCH'] = 100
	config['CURRENT_EPOCH'] = 0
	config['WORKERS'] = 4
	config['OUTPUT_DIR'] = './output/'
	config['LOG_DIR'] = ''
	config['DATA_DIR'] = ''
	config['GPUS'] = [0,1,2,3]
	config['LEARNING_RATE'] = 2.5e-5


	config['MODEL'] = dict()
	config['MODEL']['NAME'] = 'openpose_hand'
	config['MODEL']['NUM_JOINTS'] = 21
	

	config['DATASET'] = dict()
	config['DATASET']['NAME'] = 'TencentHand'

	# DATASET TRAIN related params
	config['DATASET']['TRAIN'] = dict()
	config['DATASET']['TRAIN']['BATCH_SIZE'] = 6
	config['DATASET']['TRAIN']['JSON_PATH'] = '/data/Train.json'

	# training data augmentation
	# config['DATASET['TRAIN['FLIP'] = True
	# config['DATASET['TRAIN['SCALE_FACTOR'] = 0['25
	# config['DATASET['TRAIN['ROT_FACTOR'] = 30

	# DATASET VALID related params
	config['DATASET']['VALID'] = dict()
	config['DATASET']['VALID']['BATCH_SIZE'] = 6
	config['DATASET']['VALID']['JSON_PATH'] = 'data/Valid.json'

	# optimizer related params
	# !name should be lower case for parms in parameter
	config['OPTIMIZER'] = dict()
	config['OPTIMIZER']['NAME'] = 'RMSprop'
	config['OPTIMIZER']['PARAMETERS'] = dict()
	config['OPTIMIZER']['PARAMETERS']['momentum'] = 0 
	config['OPTIMIZER']['PARAMETERS']['weight_decay'] = 0

	# optimizer shcedule related params
	# !name should be lower case for parms in parameter
	config['OPTIMIZER_SCHEDULE'] = dict()
	config['OPTIMIZER_SCHEDULE']['PARAMETERS']=dict()
	config['OPTIMIZER_SCHEDULE']['PARAMETERS']['gamma'] = 0.1
	config['OPTIMIZER_SCHEDULE']['PARAMETERS']['schedule'] = [60, 90]
	return config
def convert(x):
	x = dict(x)
	for key in x:
		if isinstance(x[key], edict):
			x[key] = convert(x[key])
	return x

if __name__ == '__main__':
    import sys
    import yaml
    config = config_sample()
    convert(config)
    print (type(config))
    yaml.dump(config,open("./config.yaml","w"), default_flow_style=False )
    # gen_config(sys.argv[1])
