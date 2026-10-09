from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results/extended';F=R/'figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
    fig.savefig(F/f'{name}.png',dpi=240);fig.savefig(F/f'{name}.pdf');fig.savefig(F/f'{name}.svg')

b=pd.read_csv(O/'broad_summary.csv'); a=pd.read_csv(O/'alpha_summary.csv'); d=pd.read_csv(O/'dependence_summary.csv'); g=pd.read_csv(O/'grid_resolution_summary.csv'); p=pd.read_csv(O/'population_phase_complexity.csv')
scens=['identical','separated','crossing','atoms']; colors=['#0072B2','#D55E00','#009E73','#CC79A7']
fig,axs=plt.subplots(2,2,figsize=(8,4.8),sharex=True,sharey=True,layout='constrained')
for ax,scenario in zip(axs.ravel(),scens):
  for band,ls in [('cp','-'),('dkw','--')]:
    q=b[(b.scenario==scenario)&(b.ratio==4)&(b.band==band)]
    ax.plot(q.n1,q.learner_cert_fraction,marker='o',ls=ls,label=band.upper())
  ax.set_xscale('log');ax.set_title(scenario);ax.grid(alpha=.2)
axs[0,0].legend();fig.supylabel('Learner certification fraction');axs[1,0].set_xlabel('Minority calibration n1');axs[1,1].set_xlabel('Minority calibration n1')
save(fig,'extended_sample_size_certification')

for col,title in zip(['rate_coverage','metric_coverage','regret_coverage'],['Rate-band coverage','Metric-band coverage','Regret-bound coverage']):
  fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained')
  for ax,scenario in zip(axs.ravel(),scens):
    q=b[(b.band=='cp')&(b.scenario==scenario)].pivot(index='ratio',columns='n1',values=col)
    im=ax.imshow(q.values,vmin=.90,vmax=1,cmap='viridis',aspect='auto');ax.set_title(scenario+' / '+title);ax.set_yticks(range(4),[f'{v:g}' for v in q.index]);ax.set_xticks(range(7),q.columns);ax.set_xlabel('Minority n1');ax.set_ylabel('n0 / n1')
    for y in range(4):
      for x in range(7):ax.text(x,y,f'{q.iloc[y,x]:.3f}',ha='center',va='center',fontsize=12,color='black' if q.iloc[y,x]>.96 else 'white')
  fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8,label='Empirical proportion (300 repetitions)')
  save(fig,'extended_cp_'+col)

fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for scenario,c in zip(scens,colors):
  for band,ls in [('cp','-'),('dkw','--')]:
    q=a[(a.scenario==scenario)&(a.band==band)]
    axs[0].plot(q.alpha,q.learner_cert_fraction,marker='o',ls=ls,color=c,label=f'{scenario} {band.upper()}')
    axs[1].plot(q.alpha,q.mean_regret_bound,marker='o',ls=ls,color=c)
axs[0].set(xlabel='alpha',ylabel='Learner certification fraction',title='Alpha sensitivity');axs[1].set(xlabel='alpha',ylabel='Mean regret upper bound',title='Conservativeness vs alpha');axs[0].legend(fontsize=10,ncol=2)
save(fig,'extended_alpha_sensitivity')

fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for scenario,c in zip(['separated','crossing'],colors[1:3]):
  for band,ls in [('cp','-'),('dkw','--')]:
    q=d[(d.scenario==scenario)&(d.band==band)]
    axs[0].plot(q.dependence,q.metric_coverage,marker='o',ls=ls,color=c,label=f'{scenario} {band.upper()}')
    axs[1].plot(q.dependence,q.learner_cert_fraction,marker='o',ls=ls,color=c,label=f'{scenario} {band.upper()}')
axs[0].set(xlabel='Shared-uniform selection probability',ylabel='Metric-band coverage',title='Dependence sensitivity');axs[1].set(xlabel='Shared-uniform selection probability',ylabel='Learner certification fraction',title='Dependence and certification');axs[0].legend(fontsize=10)
save(fig,'extended_dependence_sensitivity')

fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for scenario,c in zip(['separated','crossing'],colors[1:3]):
 q=g[g.scenario==scenario];ax.plot(q.grid_points,q.mean_bound,marker='o',color=c,label=scenario);ax.fill_between(q.grid_points,q.mean_grid_max,q.mean_bound,alpha=.12,color=c)
ax.set(xscale='log',xlabel='Prevalence grid points',ylabel='Continuum regret bound',title='Grid-resolution sensitivity');ax.legend();save(fig,'extended_grid_resolution')

fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(len(p));w=.35;ax.bar(x-w/2,p.segments,w,label='Envelope segments');ax.bar(x+w/2,p.pairwise_cut_count,w,label='All-pair cut count');ax.set_xticks(x,p.scenario);ax.set_ylabel('Count');ax.set_title('Population phase complexity across DGPs');ax.legend();save(fig,'extended_phase_complexity')
print('Wrote extended figure set')


# Candidate-pool and descriptive archive profiles use saved summaries only.
pool=pd.read_csv(O/'pool_ablation_summary.csv')
q=pool[(pool.band=='cp')&(pool.variant=='structural_unique')]
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for nt,c in zip(sorted(q.thresholds_per_score.unique()),['#0072B2','#D55E00','#009E73']):
 z=q[q.thresholds_per_score==nt];axs[0].plot(z.n1,z.strict_cert_fraction,marker='o',color=c,label=f'{nt} thresholds/score');axs[1].plot(z.n1,z.mean_bound,marker='o',color=c)
for ax in axs:ax.set_xscale('log');ax.set_xlabel('Minority calibration n1');ax.grid(alpha=.2)
axs[0].set_ylabel('Strict policy certification fraction');axs[1].set_ylabel('Mean regret upper bound');axs[0].set_title('Candidate-pool size / CP');axs[1].set_title('Pool size and bound width / CP');axs[0].legend(fontsize=10)
save(fig,'extended_pool_ablation')
a=pd.read_csv(O/'archive_resplit_primary_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for dataset in a.dataset.unique():
 z=a[(a.dataset==dataset)&(a.band=='cp')];axs[0].plot(z.calibration_fraction,z.mean_bound,marker='o',alpha=.75,label=dataset);axs[1].plot(z.calibration_fraction,z.mean_lcb_regret,marker='o',alpha=.75)
axs[0].set_ylabel('Mean source regret bound');axs[1].set_ylabel('Mean realized regret (LCB selection)')
for ax in axs:ax.set_xlabel('Calibration fraction');ax.grid(alpha=.2)
axs[0].set_title('Archive resplit / CP');axs[1].set_title('Realized regret / CP');axs[0].legend(fontsize=9,ncol=2)
save(fig,'extended_archive_resplit')

# Revised controlled pool plot: DKW M=K; no pure-multiplicity claim.
plt.rcParams.update({'font.family':'sans-serif','font.size':10,'pdf.fonttype':42})
orth=pd.read_csv(O/'orthogonal_pool_ablation_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
styles={(4,8):('#0072B2','-'),(4,32):('#0072B2','--'),(8,8):('#D55E00','-'),(8,32):('#D55E00','--')}
for band,marker in [('cp','o'),('dkw','s')]:
  for (density,k),(color,ls) in styles.items():
    z=orth[(orth.band==band)&(orth.frontier_density==density)&(orth.total_k==k)]
    axs[0].plot(z.n1,z.strict_cert_fraction,marker=marker,ls=ls,color=color,label=f'{band.upper()} R={density}, K={k}')
    axs[1].plot(z.n1,z.mean_regret_bound,marker=marker,ls=ls,color=color)
for ax in axs:ax.set_xscale('log');ax.set_xlabel('Minority calibration sample size');ax.grid(alpha=.2)
axs[0].set_ylabel('Strict policy certification fraction');axs[1].set_ylabel('Mean grid regret upper bound');axs[0].legend(fontsize=8,ncol=2)
for ext in ['png','pdf','svg']:fig.savefig(F/f'orthogonal_pool_ablation.{ext}',dpi=300)
