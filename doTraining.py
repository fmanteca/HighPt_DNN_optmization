import os, sys, stat
import random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import backend as K
from keras import callbacks
from keras import optimizers
from keras.regularizers import l2

# sys.argv[1]: nNeurons first layer
# sys.argv[2]: nhidden layers
# sys.argv[3]: learning rate 0.00005
# sys.argv[4]: batch size 1000

train = pd.read_csv('train.csv')
#train = pd.read_csv('train_fortesting.csv')

variablesTrain = ["Muon_InnerTrack_eta", "Muon_InnerTrack_phi", "Muon_InnerTrack_charge", "Muon_InnerTrack_pt",  "Muon_TunePTrack_pt", "Muon_DT_s1_nhits","Muon_DT_s1_x_mean","Muon_DT_s1_y_mean","Muon_DT_s1_z_mean","Muon_DT_s1_x_std","Muon_DT_s1_y_std","Muon_DT_s1_z_std","Muon_DT_s1_x_skew","Muon_DT_s1_y_skew","Muon_DT_s1_z_skew","Muon_DT_s1_x_kurt","Muon_DT_s1_y_kurt","Muon_DT_s1_z_kurt","Muon_DT_s2_nhits","Muon_DT_s2_x_mean","Muon_DT_s2_y_mean","Muon_DT_s2_z_mean","Muon_DT_s2_x_std","Muon_DT_s2_y_std","Muon_DT_s2_z_std","Muon_DT_s2_x_skew","Muon_DT_s2_y_skew","Muon_DT_s2_z_skew","Muon_DT_s2_x_kurt","Muon_DT_s2_y_kurt","Muon_DT_s2_z_kurt","Muon_DT_s3_nhits","Muon_DT_s3_x_mean","Muon_DT_s3_y_mean","Muon_DT_s3_z_mean","Muon_DT_s3_x_std","Muon_DT_s3_y_std","Muon_DT_s3_z_std","Muon_DT_s3_x_skew","Muon_DT_s3_y_skew","Muon_DT_s3_z_skew","Muon_DT_s3_x_kurt","Muon_DT_s3_y_kurt","Muon_DT_s3_z_kurt","Muon_DT_s4_nhits","Muon_DT_s4_x_mean","Muon_DT_s4_y_mean","Muon_DT_s4_x_std","Muon_DT_s4_y_std","Muon_DT_s4_x_skew","Muon_DT_s4_y_skew","Muon_DT_s4_x_kurt","Muon_DT_s4_y_kurt"]

genpT = ["Muon_Genpt"]

K.clear_session()

model = models.Sequential()

# First layer:
model.add(layers.Dense(int(sys.argv[1]), activation='relu',input_dim=len(variablesTrain))) #, kernel_regularizer=l2(0.001)

# Hiden layers:
for i in range(0,int(sys.argv[2])):
    if i == 0:
        currentNeurons = int(sys.argv[1])

    if random.choice([True, False]):
        model.add(layers.Dense(currentNeurons, activation='relu'))
    else:
        currentNeurons = currentNeurons/2
        model.add(layers.Dense(currentNeurons, activation='relu'))
        if currentNeurons == 2:
            break

# Last layer
model.add(layers.Dense(1, activation='linear'))

opt = optimizers.Adam(float(sys.argv[3]))
model.compile(loss="mean_squared_error", optimizer=opt)


history = model.fit(train[variablesTrain],train[genpT],validation_split=0.1, epochs=1000, batch_size=int(sys.argv[4]), verbose=1, callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=50,verbose=1)])


# Save the model

model.save('model_nFirstNeurons' + sys.argv[1] + '_nHiddenLayers_' + sys.argv[2] + '_LearningRate_' + sys.argv[3] + '_BatchSize_' + sys.argv[4] + '.h5')

# # Summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.ylabel('MSE',fontsize=10)
# plt.xlabel('Epoch',fontsize=10)
# plt.legend(['train', 'validation'], loc='upper right',fontsize=11)
# plt.tick_params(axis='x', labelsize=10)
# plt.tick_params(axis='y', labelsize=7)
# plt.yscale('log')


# plt.savefig('history/model_loss_nFirstNeurons' + sys.argv[1] + '_nHiddenLayers_' + sys.argv[2] + '_LearningRate_' + sys.argv[3] + '_BatchSize_' + sys.argv[4] + '.png')


