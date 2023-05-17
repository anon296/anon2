import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def post_proc_df(df):
    df = df[~df.isna().any(axis=1)]
    df.index = df['img_label']
    df = df[['total']+[f'level{i}' for i in range(4)]]
    df.index=[f'fract-dim {2-float(x[-1])/10:.1f}' for x in df.index]
    return df.sort_index()

rename_dict = {'patch_ent':'total'}
rename_dict.update({f'patch_ent{i}':f'level{i}' for i in range(4)})
all_dfs = []
for i in range(5):
    fpath = f'fract{i}_results.csv'
    df = pd.read_csv(fpath,index_col=0)
    if 'patch_ent' in df.columns:
        df = df.rename(rename_dict,axis=1)
    post_procced_df = post_proc_df(df)
    all_dfs.append(post_procced_df)

means_df = pd.DataFrame(np.stack([df.to_numpy() for df in all_dfs]).mean(axis=0),index=post_procced_df.index,columns=post_procced_df.columns)
stds_df = pd.DataFrame(np.stack([df.to_numpy() for df in all_dfs]).std(axis=0),index=post_procced_df.index,columns=post_procced_df.columns)
means_df.to_csv('means_fract_results.csv')
stds_df.to_csv('stds_fract_results.csv')

means_and_stds_list_of_lists = [[f'{m:.2f} ({s:.2f})' for m,s in zip(means_row.values,stds_row.values)] for (i,means_row),(i,stds_row) in zip(means_df.iterrows(),stds_df.iterrows())]
boths_df = pd.DataFrame(means_and_stds_list_of_lists,index=post_procced_df.index,columns=post_procced_df.columns)
boths_df.to_latex('meanstd_fract_results.csv')

for i in range(1,9):
    #plt.plot(np.linspace(1.1,1.9,9),means_df[f'level{i}'],mm
    means = means_df.loc[f'fract-dim 1.{i}'].drop('total')
    stds = stds_df.loc[f'fract-dim 1.{i}'].drop('total')
    plt.plot(np.arange(1,5),means,label=f'fract-dim 1.{i}')
    plt.xticks(np.arange(1,5))
    plt.fill_between(np.arange(1,5),means-stds,means+stds,alpha=0.4)
    plt.legend()
    plt.xlabel('Level')
    plt.ylabel('Complexity Score')
plt.savefig('fract_results_as_plot.png')
