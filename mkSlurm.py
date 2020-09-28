import os, sys, stat
import ROOT as r
from array import array
import numpy as np
import random

# python mkSlurm.py /gpfs/users/mantecap/CMSSW_11_1_0/src /gpfs/users/mantecap/DNN_optimization

templateSlurm = """#!/bin/bash
cd CMSSWRELEASE
eval `scramv1 runtime -sh`
cd WORKINGPATH
python doTraining.py NNEURONS NLAYERS LEARNINGRATE BATCH
"""

########################## Main program #####################################
if __name__ == "__main__":

    cmsswRelease = sys.argv[1]
    workingpath = sys.argv[2]

    for i in range(0,1000):
        
        neurons = random.choice([128,256,512,1024])
        layers = random.randint(5, 15)
        lr = random.uniform(0, 0.001)
        batch = random.choice([256,512,1024,2048])

        template = templateSlurm
        template = template.replace('CMSSWRELEASE', cmsswRelease)
        template = template.replace('WORKINGPATH', workingpath) 
        template = template.replace('NNEURONS', str(neurons))
        template = template.replace('NLAYERS', str(layers))
        template = template.replace('LEARNINGRATE', str(lr))
        template = template.replace('BATCH', str(batch))
            
        f = open('send_nFirstNeurons_' + str(neurons) + '_nHiddenLayers_' + str(layers) + '_LearningRate_' + str(lr) + '_BatchSize_' + str(batch) + '.sh', 'w')
        f.write(template)
        f.close()
        os.chmod('send_nFirstNeurons_' + str(neurons) + '_nHiddenLayers_' + str(layers) + '_LearningRate_' + str(lr) + '_BatchSize_' + str(batch) + '.sh', 0755)     
    













     

