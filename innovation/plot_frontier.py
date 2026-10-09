"""Plot exactly the policies generating the full simulation envelope."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from policy_certificate import macro_f1
R=Path(__file__).resolve().parent
(R/'figures').mkdir(parents=True,exist_ok=True)
d=json.loads((R/'results/simulation_metadata.json').read_text())['dgp']['crossing']
a=np.array(d['population_tpr']); b=np.array(d['population_fpr'])
phase=d['population_phase'];ids=list(dict.fromkeys(s['winners'][0] for s in phase['segments']))
p=np.linspace(.02,.5,1201); curves=macro_f1(a[:,None],b[:,None],p[None,:])
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
for i,color in zip(ids,colors):
    label=f'Score {i//12+1}, t={i%12/10:.1f}'
    ax[0].scatter(b[i],a[i],s=65,color=color,label=label)
    ax[0].annotate(f'{i+1}',(b[i],a[i]),xytext=(6,-3),textcoords='offset points')
    ax[1].plot(p,curves[i],color=color,lw=1.8)
ax[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='A  Envelope policies')
ax[0].legend(loc='upper left',fontsize=9)
ax[0].set_xlim(0,.55);ax[0].set_ylim(0,1)
ax[1].plot(p,curves.max(axis=0),color='black',ls=':',lw=2,label='Full-pool envelope')
for bound in phase['boundaries'][1:-1]:
    q=bound['p'];w=bound['winners'];v=macro_f1(a[w],b[w],q)
    assert np.ptp(v)<1e-8
    assert abs(v[0]-macro_f1(a,b,q).max())<1e-8
    ax[1].axvline(q,color='0.5',ls='--',lw=.8)
ax[1].set(xlabel='Prevalence',ylabel='Macro-F1',title='B  Curves and envelope switches')
ax[1].legend(fontsize=9)
for ext in ['png','pdf','svg']:
    fig.savefig(R/'figures'/f'frontier_single_crossing.{ext}',dpi=220)
(R/'figures/frontier_data.json').write_text(json.dumps({'policy_indices_zero_based':ids,'tpr':a[ids].tolist(),'fpr':b[ids].tolist(),'phase':phase},indent=2))
print('Plotted policies',ids,'with verified boundary equalities')
