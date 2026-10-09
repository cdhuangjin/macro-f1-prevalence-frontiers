"""Rebuild editable result figures from the archived derived tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
(R/'figures').mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(R/'figures'/f'{name}.{ext}',dpi=220)
sim=pd.read_csv(R/'results/simulation_summary.csv');sub=sim[sim.band=='cp']
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,col,title in zip(axes,['learner_certification_grid_fraction','mean_regret_bound'],['Learner certification fraction (CP)','Mean regret upper bound (CP)']):
    table=sub.pivot(index='scenario',columns='n1',values=col).reindex(['atoms','crossing','identical','separated'])
    ax.imshow(table.values,cmap='viridis',aspect='auto',vmin=0,vmax=1 if col.startswith('learner') else None)
    ax.set_xticks(range(3),table.columns);ax.set_yticks(range(4),table.index)
    ax.set_xlabel('Minority calibration n1');ax.set_title(title)
    for y in range(4):
        for x in range(3):ax.text(x,y,f'{table.iloc[y,x]:.3f}',ha='center',va='center',color='white' if table.iloc[y,x]<.25 else 'black')
save(fig,'certificate_simulation_heatmap')
phases=json.loads((R/'results/archive_phases.json').read_text())
counts={}
for ph in phases:
    if ph['forest_seed']==101:counts[ph['dataset']]=counts.get(ph['dataset'],0)+(len(ph['segments'])>1)
assert sum(counts.values())==102
df=pd.read_csv(R/'results/archive_application.csv');df=df[(df.forest_seed==101)&(df.prevalence_position=='source')]
means=df.groupby('band')[['evaluation_lcb_regret','regret_bound']].mean().reindex(['cp','dkw'])
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
names=sorted(counts);axes[0].bar(names,[counts[n] for n in names],color='#4477AA')
axes[0].tick_params(axis='x',labelrotation=45,labelsize=9)
for label in axes[0].get_xticklabels():label.set_ha('right')
axes[0].set(title='Archived prevalence sensitivity (seed 101)',ylabel='Blocks with a policy switch',ylim=(0,15))
x=np.arange(2);axes[1].bar(x-.18,means.evaluation_lcb_regret,.36,label='Realized evaluation regret',color='#77AA55');axes[1].bar(x+.18,means.regret_bound,.36,label='Regret upper bound',color='#EE8833')
axes[1].set_xticks(x,['CP','DKW']);axes[1].set_title('Selection at source prevalence');axes[1].legend(fontsize=9)
save(fig,'archive_sensitivity_summary')
print('Rebuilt two result figures with SVG/PDF and source tables')
