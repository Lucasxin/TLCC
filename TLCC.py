import os
from scipy.stats import pearsonr
import numpy as np
import argparse
import pickle
import seaborn as sns
from scipy.signal import hilbert, butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import plotly.express as px
from tqdm import tqdm
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))
data = np.load('original_data.npz')['data']
data=np.transpose(data,(1,0,2))
print(data.shape)
num_nodes= data.shape[1]
d1 = pd.Series(data[0:14400, 3, 0].flatten())
d2 = pd.Series(data[0:14400, 34 , 0].flatten())
seconds = 5
fps = 30
rs = [crosscorr(d1,d2, lag) for lag in range(-int(72),int(72))]
print(rs)
offset = np.floor(len(rs)/2)-np.argmax(rs) #位置差：中线位置——最大值位置
f,ax=plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} frames\nRoad1 leads <> Road2 leads',ylim=[-0.2,1.0],xlim=[0,144], xlabel='Offset',ylabel='Pearson r')
ax.set_xticklabels([int(item-72) for item in ax.get_xticks()]);
plt.legend()
plt.show()
# Plotly interactive plot
# 加窗的时间滞后互相关
seconds = 5
fps = 30
no_splits = 10
#samples_per_split = d1.shape[0]/no_splits
samples_per_split = 144
rss=[]

for i in range(0, 28):
    
    d3 = d1[ i * 144 : (i+1) * 144]
    d4 = d2[ i *144 : (i+1) * 144]
    rs = [crosscorr(d3,d4, lag) for lag in range(-int(72),int(72))]
    rss.append(rs)
#print(rss)
rss = pd.DataFrame(rss)
#print(rss)
f,ax = plt.subplots(figsize=(10,5))
sns.heatmap(rss,cmap='RdBu_r',ax=ax)
ax.set(title=f'Windowed Time Lagged Cross Correlation',xlim=[0,144], xlabel='Offset',ylabel='Window epochs')
ax.set_xticklabels([int(item-72) for item in ax.get_xticks()]);

plt.show()

Mask = np.zeros((num_nodes, num_nodes))
for i in tqdm (range(num_nodes)):
    for j in range (i, num_nodes):
        noffset =[]
        nors=[]
        d1 = pd.Series(data[0:4032, i, 0].flatten())
        d2 = pd.Series(data[0:4032, j , 0].flatten())
        for n in range(0, 28):
            d3 = d1[ n * 144 : (n+1) * 144]
            d4 = d2[ n *144 : (n+1) * 144]
            rs = [crosscorr(d3,d4, lag) for lag in range(-int(72),int(72))]
            offset = np.floor(len(rs)/2)-np.argmax(rs) #负数的时候是s2-lead，整数的时候是s1-lead
            noffset.append(offset)
        #print(noffset)
        avg = np.average (noffset)
        avg_pr = np.average (rs)
        #print(avg_pr)
        if (avg >= 0): 
            Mask[i][j]= 1
        if (avg < 0):
            Mask[j][i]= 1
    
np.save("TLCC.npy",Mask)
with open("TLCC.pkl", 'wb') as f: # Save to pickle file.
        pickle.dump(Mask, f, protocol=2)
