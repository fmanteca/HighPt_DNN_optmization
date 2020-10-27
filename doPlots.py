from keras import models
from scipy import stats


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model = models.load_model('model_nFirstNeurons256_nHiddenLayers_13_LearningRate_0.000492406500248_BatchSize_1024.h5')
variablesTrain = ["Muon_InnerTrack_eta", "Muon_InnerTrack_phi", "Muon_InnerTrack_charge", "Muon_InnerTrack_pt",  "Muon_TunePTrack_pt", "Muon_DT_s1_nhits","Muon_DT_s1_x_mean","Muon_DT_s1_y_mean","Muon_DT_s1_z_mean","Muon_DT_s1_x_std","Muon_DT_s1_y_std","Muon_DT_s1_z_std","Muon_DT_s1_x_skew","Muon_DT_s1_y_skew","Muon_DT_s1_z_skew","Muon_DT_s1_x_kurt","Muon_DT_s1_y_kurt","Muon_DT_s1_z_kurt","Muon_DT_s2_nhits","Muon_DT_s2_x_mean","Muon_DT_s2_y_mean","Muon_DT_s2_z_mean","Muon_DT_s2_x_std","Muon_DT_s2_y_std","Muon_DT_s2_z_std","Muon_DT_s2_x_skew","Muon_DT_s2_y_skew","Muon_DT_s2_z_skew","Muon_DT_s2_x_kurt","Muon_DT_s2_y_kurt","Muon_DT_s2_z_kurt","Muon_DT_s3_nhits","Muon_DT_s3_x_mean","Muon_DT_s3_y_mean","Muon_DT_s3_z_mean","Muon_DT_s3_x_std","Muon_DT_s3_y_std","Muon_DT_s3_z_std","Muon_DT_s3_x_skew","Muon_DT_s3_y_skew","Muon_DT_s3_z_skew","Muon_DT_s3_x_kurt","Muon_DT_s3_y_kurt","Muon_DT_s3_z_kurt","Muon_DT_s4_nhits","Muon_DT_s4_x_mean","Muon_DT_s4_y_mean","Muon_DT_s4_x_std","Muon_DT_s4_y_std","Muon_DT_s4_x_skew","Muon_DT_s4_y_skew","Muon_DT_s4_x_kurt","Muon_DT_s4_y_kurt"]



test = pd.read_csv('test.csv')





Rreco = abs(test.Muon_Genpt.values-test.Muon_TunePTrack_pt.values)/test.Muon_Genpt.values
Rpred = abs(test.Muon_Genpt.values-model.predict(test[variablesTrain]).ravel())/test.Muon_Genpt
pTpred = model.predict(test[variablesTrain]).ravel()
tunePpT = test.Muon_TunePTrack_pt.values
genpT = test.Muon_Genpt.values
RelRreco = np.std(abs(test.Muon_Genpt.values-test.Muon_TunePTrack_pt.values)/test.Muon_Genpt.values)
RelPred = np.std(abs(test.Muon_Genpt.values-model.predict(test[variablesTrain]).ravel())/test.Muon_Genpt)



plt.figure(figsize=(15,10)) 
plt.hist2d(genpT, tunePpT, bins=[50,50] ,range=[[200, 2500], [200, 5000]], norm=mpl.colors.LogNorm())
plt.xlabel('GenpT [GeV]',fontsize=14)
plt.ylabel('TuneP_pT [GeV]',fontsize=14)
plt.tick_params(axis='both', labelsize=13)

#legend
clb = plt.colorbar()
clb.set_label('nMuons', fontsize=15)
clb.ax.tick_params(labelsize=13)

plt.savefig('data_test_tuneppt_genpt.png')
plt.clf()
plt.cla()


plt.figure(figsize=(15,10)) 
plt.hist2d(genpT, pTpred, bins=[50,50] ,range=[[200, 2500], [200, 5000]], norm=mpl.colors.LogNorm())
plt.xlabel('GenpT [GeV]',fontsize=14)
plt.ylabel('Predicted pT [GeV]',fontsize=14)
plt.tick_params(axis='both', labelsize=13)


#legend
clb = plt.colorbar()
clb.set_label('nMuons', fontsize=15)
clb.ax.tick_params(labelsize=13)


plt.savefig('data_test_ptpred_genpt.png')
plt.clf()
plt.cla()



print('TuneP R mean pT[1200,2000]:',np.mean(abs(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values-test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_TunePTrack_pt.values)/test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values), '+/-', np.std(abs(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values-test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_TunePTrack_pt.values)/test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values)/np.sqrt(len(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values)))
print('Prediced R mean: ', np.mean(abs(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values-model.predict(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)][variablesTrain]).ravel())/test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values), '+/-', np.std(abs(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values-model.predict(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)][variablesTrain]).ravel())/test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values)/np.sqrt(len(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values)))


print('TuneP R std pT[1200,2000]:',np.std(abs(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values-test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_TunePTrack_pt.values)/test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values))
print('Prediced R std: ', np.std(abs(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values-model.predict(test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)][variablesTrain]).ravel())/test[(test.Muon_Genpt>1200) & (test.Muon_Genpt<2000)].Muon_Genpt.values))



ptbins = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2250,2500]


x = []
ex=[]
Relreco = []
Relpred = []
Biasreco = []
eBiasreco = []
Biaspred = []
eBiaspred = []

for i in list(zip(ptbins, ptbins[1:] + ptbins[:1]))[:-1]:
    x.append((i[0]+i[1])/2)
    ex.append((i[1]-i[0])/2)
    Relreco.append(np.std(abs(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values-test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_TunePTrack_pt.values)/test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values))
    Relpred.append(np.std(abs(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values-model.predict(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])][variablesTrain]).ravel())/test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values))
    Biasreco.append(np.mean(abs(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values-test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_TunePTrack_pt.values)/test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values))
    eBiasreco.append(np.std(abs(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values-test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_TunePTrack_pt.values)/test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values)/np.sqrt(len(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values)))
    Biaspred.append(np.mean(abs(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values-model.predict(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])][variablesTrain]).ravel())/test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values))
    eBiaspred.append(np.std(abs(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values-model.predict(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])][variablesTrain]).ravel())/test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values)/np.sqrt(len(test[(test.Muon_Genpt>i[0]) & (test.Muon_Genpt<i[1])].Muon_Genpt.values)))


plt.errorbar(x, Relreco, xerr=ex , fmt='o')
plt.errorbar(x, Relpred, xerr=ex , fmt='o')

plt.xlabel('Muon GenpT [GeV]',fontsize=13)
plt.ylabel('$\sigma_{R}$',fontsize=13)
plt.tick_params(axis='both', labelsize=13)
plt.legend(['TuneP', 'DNN'], loc='upper left')
plt.savefig('SigmaR_vs_genpT.png')

plt.clf()
plt.cla()

plt.errorbar(x, Biasreco, yerr=eBiasreco, xerr=ex , fmt='o')
plt.errorbar(x, Biaspred, yerr=eBiaspred, xerr=ex , fmt='o')


plt.xlabel('Muon GenpT [GeV]',fontsize=13)
plt.ylabel('$\mu_{R}$',fontsize=13)
plt.tick_params(axis='both', labelsize=13)
plt.legend(['TuneP', 'DNN'], loc='upper left')
plt.savefig('MeanR_vs_genpT.png')

plt.clf()
plt.cla()
