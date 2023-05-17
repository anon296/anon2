import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#dset_groups = {'Real-world Images': (['im','cifar','dtd'],'complex_dsets'),
                #'MNIST and Synthetic Images':(['mnist','stripes','halves'],'simp_dsets')}
dset_groups = {'Real-world Images': (['im','cifar','dtd'],'complex_dsets')}


dsets = [d for dg in dset_groups.values() for d in dg[0]]
exp_name = 'main_run4'

max_val = pd.concat([pd.read_csv(f'experiments/main_run4/{d}/{d}_results.csv',index_col=0).drop(['means','stds'],axis=0) for d in dsets],keys=dsets,axis=0)['level 4'].max()

for group_name,(dset_group,code_name) in dset_groups.items():
    for dset in dset_group:
        df_dset = pd.read_csv(f'experiments/{exp_name}/{dset}/{dset}_results.csv',index_col=0)[[f'level {i}' for i in range(1,5)]]
        #full_dset_name = 'imagenet' if dset=='im' else 'dtd2' if dset=='dtd' else dset
        full_dset_name = 'horn' if dset=='im' else 'fabric' if dset=='dtd' else 'cat'
        means = df_dset.loc['means'].to_numpy()/max_val
        errs = df_dset.loc['stds'].to_numpy()/(max_val*5)
        #plt.errorbar(np.arange(4),means,yerr=errs,label=dset)
        #x = np.array([4,8,16,32])
        x = np.arange(4)
        plt.plot(x,means,label=full_dset_name)
        plt.xticks(x)
        plt.ylim(bottom=0.,top=1.1)
        plt.legend()
        #plt.fill_between(np.arange(4),means-errs/2,means+errs/2,color='#90dcff')
        plt.fill_between(x,means-errs/2,means+errs/2,alpha=0.4)
    #plt.xscale('log',base=2)
    plt.xlabel('Scale')
    plt.ylabel('Complexity Score')
    plt.title(group_name)
    #plt.savefig(f'experiments/{exp_name}/{code_name}_scale_plots_sample_stds.png')
    plt.savefig('jim.png')
    plt.show()
    plt.clf()
