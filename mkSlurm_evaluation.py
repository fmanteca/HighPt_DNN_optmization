import os, sys, stat
import ROOT as r
from array import array
import numpy as np
import random

# python mkSlurm_evaluation.py /gpfs/users/mantecap/CMSSW_11_1_0/src /gpfs/users/mantecap/DNN_optimization

templateSlurm = """#!/bin/bash
cd CMSSWRELEASE
eval `scramv1 runtime -sh`
cd WORKINGPATH
python doEvaluation.py FILE
"""

########################## Main program #####################################
if __name__ == "__main__":

    cmsswRelease = sys.argv[1]
    workingpath = sys.argv[2]

    for imodel in glob.glob("*h5*"):

        template = templateSlurm
        template = template.replace('CMSSWRELEASE', cmsswRelease)
        template = template.replace('WORKINGPATH', workingpath) 
        template = template.replace('FILE', str(imodel))

        f = open('send_doEval_' + str(imodel)  + ' .sh', 'w')
        f.write(template)
        f.close()
        os.chmod('send_doEval_' + str(imodel)  + ' .sh', 'w', 0755)     
    













     

