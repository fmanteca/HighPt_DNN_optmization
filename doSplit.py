import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 

files = ['TrainFile_minus.csv', 'TrainFile_plus.csv']
mylist = []
    
for filename in files:
#    df = pd.read_csv(filename)
    mylist.append(pd.read_csv(filename))
#    del df

data = pd.concat(mylist, ignore_index=True)
del mylist

data = data[(abs(data.Muon_InnerTrack_eta)<0.9) & (data.Muon_Genpt>200.)]


train, test = train_test_split(data, test_size=0.2)

del data

train.to_csv('train.csv')
test.to_csv('test.csv')
