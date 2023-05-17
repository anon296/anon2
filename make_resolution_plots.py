import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




for size in [32,64,128]:
    y =pd.read_csv(f'experiments/downsample{size}/im/im_results.csv',index_col=0).loc['means'].iloc[3:].values.astype(float)
    errs =pd.read_csv(f'experiments/downsample{size}/im/im_results.csv',index_col=0).loc['stds'].iloc[3:].values.astype(float)
    errs = (errs**0.5)/5
    plt.plot(np.arange(4),y,label=f'res{size}')
    plt.ylim(bottom=0.,top=1.)
    plt.fill_between(np.arange(4),y-errs,y+errs,alpha=0.4)

# get full resolution
means = pd.read_csv('experiments/main_run/im/im_results.csv',index_col=0).iloc[-4:,0]
errs = pd.read_csv('experiments/main_run/im/im_results.csv',index_col=0).iloc[-4:,2]
errs = (errs**0.5)/5
plt.plot(np.arange(4),y,label=f'full-res')
plt.fill_between(np.arange(4),y-errs,y+errs,alpha=0.4)
plt.legend()
plt.xlabel('Scale')
plt.ylabel('Complexity Score')

plt.savefig(f'experiments/resolution_lowered.png')
plt.show()
