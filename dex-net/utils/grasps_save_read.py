#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : MrRen-sdhm
# File Name  : grasps_save_read.py

import numpy as np
import glob
import os
import pickle


def grasps_save(grasps, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(grasps, f)


def grasps_read(filename):
    grasps = pickle.load(open(filename, 'rb'))
    if isinstance(grasps, list):
        return grasps
    else:
        return [grasps]


if __name__ == '__main__':
    path = os.environ['HOME'] + "/grasp.pickle"
    grasps = grasps_read(path)
    print(grasps)
