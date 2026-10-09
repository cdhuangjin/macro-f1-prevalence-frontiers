"""Plot recorded complete-function timing results; timings are machine-dependent."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
F=R/'figures';F.mkdir(parents=True,exist_ok=True)
data=pd.read_csv(R/'results/runtime/runtime_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,family in zip(axes,['sqrt','diagonal']):
    for method,label in [('stack_complete','Stack (complete function)'),('all_pairs_complete','All pairs (complete function)')]:
        sub=data[(data.family==family)&(data.method==method)].sort_values('K')
        ax.plot(sub.K,sub.seconds_median,marker='o',label=label)
        ax.fill_between(sub.K,sub.seconds_min,sub.seconds_max,alpha=.15)
    ax.set(xscale='log',yscale='log',xlabel='Candidate policies K',ylabel='Runtime (seconds)',title=family)
    ax.legend(fontsize=8);ax.grid(alpha=.2)
for ext in ['png','pdf','svg']:fig.savefig(F/f'envelope_runtime.{ext}',dpi=220)
