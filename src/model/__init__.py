# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .weakly import Weakly
from .BaseModel import BaseModel
from .ICCV17 import CPMWeakly
from .OpenPose import OpenPose
from .Weakly_direct_regression import Weakly_direct_regression
from .Hand25D import Hand25D
from .two_stream import two_stream
from .depth_regularizer import depth_regularizer
from .direct_two_stream import direct_two_stream
from .Weakly_direct_regression_with_depth import Weakly_direct_regression_with_depth