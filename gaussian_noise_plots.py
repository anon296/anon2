import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


max_complexity = 35 # guesstimate, need to compute this properly for a full experiment
dsets = ['im','cifar','dtd']
for dset_name in dsets:
    dfs = [pd.read_csv(f'experiments/gaussian_{n}percent/{dset_name}/{dset_name}_results.csv',index_col=0) for n in [0,5,10,15,20]]
    full_dset_name = 'imagenet' if dset_name=='im' else 'dtd2' if dset_name=='dtd' else dset_name
    #means, errs = np.array(dfs_dset['mean']), np.array(dfs_dset['std'])
    means = np.array([df.loc['patch_ent','mean'] for df in dfs])/max_complexity
    errs = np.array([df.loc['patch_ent','std'] for df in dfs])/max_complexity
    #plt.errorbar(np.arange(4),means,yerr=errs,label=dset)
    x = 5*np.arange(5)
    plt.plot(x,means,label=full_dset_name)
    plt.xticks(x)
    #plt.fill_between(np.arange(4),means-errs/2,means+errs/2,color='#90dcff')
    plt.fill_between(x,means-errs/5,means+errs/5,alpha=0.4)
plt.legend()
plt.xlabel('Percentage Gaussian noise added')
plt.ylabel('Complexity Score')

plt.savefig(f'experiments/gaussian_noises.png')
