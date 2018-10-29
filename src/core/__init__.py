# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..","utils"))

from .main import train, validate
