#!/bin/bash
cd /gpfs/users/mantecap/CMSSW_11_1_0/src
eval `scramv1 runtime -sh`
cd /gpfs/users/mantecap/DNN_optimization
python doPlots.py
