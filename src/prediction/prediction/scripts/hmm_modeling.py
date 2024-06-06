import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.io as sio
import copy
from hmmlearn.hmm import GMMHMM, GaussianHMM

def making_dataset(sequence):

    X = []
    Y = []
    CHECK = []
    # [station, d, x,y, heading, v, mode, length, width]
    for i in range(11, len(sequence)):

        temp = np.concatenate([np.abs(sequence[i-10:i,1][:,np.newaxis]), 10*np.abs(sequence[i-10:i,1]-sequence[i-11:i-1,1])[:,np.newaxis]], axis=-1)

        check = np.concatenate([sequence[i-10:i, :],sequence[i-10:i,:2]-sequence[i-11:i-1,:2]], axis=-1)

        X.append(temp)
        Y.append(sequence[i-10:i,-3])
        CHECK.append(check)

    return X, Y, CHECK


FileList = glob.glob("/home/mmc_ubuntu/Work/programmers/2023/HMM_modeling/*_rev.pickle")
coord_init = [326130, 340830]

print(FileList)    
    
XX, YY, CHECK = [], [], []

for i in range(1): #(len(FileList)):
    
    with open(FileList[-1], "rb") as f:
        print(FileList[-1])
        # [station, d, x,y, heading, v, mode, length, width]
        data = pickle.load(f)

    for key in data.keys():
        
              
        LC = (np.where(data[key][:,-3]==1))[0]
        

        
        X,Y, check = making_dataset(data[key])
        

        XX += X
        YY += Y
        
        CHECK +=check

n_state = 2


np.random.seed(0)
model_LC = GaussianHMM(n_components=2,
               min_covar = 0.1,
               startprob_prior=np.array([0.5,0.5]),
               transmat_prior=np.array([[0.8,0.2],[0.2, 0.8]]),
               n_iter = 100)

model_LK = GaussianHMM(n_components=2,
               min_covar = 0.1,
               startprob_prior=np.array([0.5,0.5]),
               transmat_prior=np.array([[0.8,0.2],[0.2, 0.8]]),
               n_iter = 100)


XX = np.array(XX)
YY = np.array(YY)


LC_idx = np.where((YY[:,-2]==1)&(YY[:,-1]==1))[0]
LK_idx = np.where((YY[:,-2]==0)&(YY[:,-1]==0))[0]

# sample_idx_list = np.random.randint(0,len(test_idx),10)

XX_LC, XX_LK = XX[LC_idx], XX[LK_idx]
YY_LC, YY_LK = YY[LC_idx], YY[LK_idx]




length_LC = np.array([[10] for i in range(len(XX_LC))])
length_LK = np.array([[10] for i in range(len(XX_LK))])

XX_LC.resize(len(XX_LC)*10,n_state)
XX_LK.resize(len(XX_LK)*10,n_state)

# print(XX_LC.shape)

hidden_states_LC = model_LC.fit(X=XX_LC, lengths=length_LC)
hidden_states_LK = model_LK.fit(X=XX_LK, lengths=length_LK)



XX_re = XX.reshape(-1,10,n_state)


test_idx = np.where((YY[:,0]==0) &(YY[:,-1]==1))[0]
test_idx = np.where((YY[:,0]-YY[:,-1])>0)[0]

sample_idx_list = np.random.randint(0,len(test_idx),10)

for sample_idx in sample_idx_list:

    # print(model.predict(XX_re[sample_idx]))
    print(XX_re[test_idx[sample_idx]])
    # print(model.decode(XX_re[sample_idx]))
    print(model_LC.score(XX_re[test_idx[sample_idx]], lengths=[10]), model_LK.score(XX_re[test_idx[sample_idx]],lengths=[10]))
    
    print(YY[test_idx[sample_idx]])

LK_True = []
LK_False = []
LC_True = []
LC_False = []


for sample in range(len(XX)):
    LCs, LKs = model_LC.score(XX_re[sample], lengths=[10]), model_LK.score(XX_re[sample],lengths=[10])
    
    if YY[sample,-1] == 1:
        if LCs > LKs:
            LC_True.append(sample)
        else:
            LC_False.append(sample)
            
    else:
        if LCs < LKs:
            LK_True.append(sample)
        else:
            LK_False.append(sample)



# with open("./model_LC.pickle", 'wb') as f:
#     pickle.dump(hidden_states_LC, f)
    
# with open("./model_LK.pickle", 'wb') as f:
#     pickle.dump(hidden_states_LK, f)



with open("./model_LC.pickle", 'rb') as f:
    hhc = pickle.load(f)
    
with open("./model_LK.pickle", 'rb') as f:
    hhk = pickle.load(f)
    

LK_True = []
LK_False = []
LC_True = []
LC_False = []

    
 
for sample in range(len(XX)):
    LCs, LKs = hhc.score(XX_re[sample], lengths=[10]), hhk.score(XX_re[sample],lengths=[10])
    
    if YY[sample,-1] == 1:
        if LCs > LKs:
            LC_True.append(sample)
        else:
            LC_False.append(sample)
            
    else:
        if LCs < LKs:
            LK_True.append(sample)
        else:
            LK_False.append(sample)
         

print("LC True : ", len(LC_True)/(len(LC_True)+len(LC_False)))

print("LK True : ", len(LK_True)/(len(LK_True)+len(LK_False)))

print("Total : ", (len(LC_True)+len(LK_True))/(len(LC_True)+len(LC_False)+len(LK_True)+len(LK_False)))
