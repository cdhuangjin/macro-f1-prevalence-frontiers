"""Expanded, reproducible experiment battery for contribution C.

This is additive to the original 1,000-repetition study. It varies minority
sample size, class-imbalance ratio, alpha, score dependence and prevalence-grid
resolution. Every row is an actual Monte Carlo cell summary; no synthetic
summary rows are written.
"""
from __future__ import annotations
import json,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betaincinv,betainc
from policy_certificate import macro_f1,bounds,recommendations,phase_diagram,interval_regret_bound
from run_development import DGPS,population_rates

R=Path(__file__).resolve().parent; OUT=R/'results/extended';OUT.mkdir(parents=True,exist_ok=True)
TS=np.r_[np.linspace(0,1,11),np.inf]; PS=np.linspace(.02,.5,25)
CONSTANTS=np.tile(np.r_[1,np.zeros(10),-1],3); GROUPS=np.repeat(np.arange(3),len(TS))

def sample_scores_dep(rng,n,param,shared=False,atoms=False,dependence=.7):
    common=rng.uniform(size=(n,1))
    if shared:
        u=np.repeat(common,3,axis=1)
    else:
        independent=rng.uniform(size=(n,3)); choose=rng.uniform(size=(n,3))<dependence
        u=np.where(choose,common,independent)
    x=np.column_stack([betaincinv(a,b,u[:,m]) for m,(a,b) in enumerate(param)])
    return np.floor(4*x+.5)/4 if atoms else x

def one_cell(name,dgp,n1,n0,alpha,dependence,reps,seed):
    a=population_rates(dgp['pos'],TS,dgp.get('atoms',False));b=population_rates(dgp['neg'],TS,dgp.get('atoms',False))
    truth=macro_f1(a[:,None],b[:,None],PS[None,:]);oracle=truth.max(axis=0)
    rows=[];rng=np.random.default_rng(seed)
    for rep in range(reps):
        pos=sample_scores_dep(rng,n1,dgp['pos'],dgp.get('shared',False),dgp.get('atoms',False),dependence)
        neg=sample_scores_dep(rng,n0,dgp['neg'],dgp.get('shared',False),dgp.get('atoms',False),dependence)
        ah=np.concatenate([(pos[:,m,None]>=TS).mean(axis=0) for m in range(3)])
        bh=np.concatenate([(neg[:,m,None]>=TS).mean(axis=0) for m in range(3)])
        for kind in ('cp','dkw'):
            band=bounds(ah,bh,n1,n0,alpha=alpha,kind=kind,n_models=3,constants=CONSTANTS)
            rec=recommendations(ah,bh,band,PS,GROUPS);idx=np.arange(len(PS));selected=rec['selected']
            actual=oracle-truth[selected,idx]
            al,au,bl,bu=band
            rows.append({'scenario':name,'n1':n1,'n0':n0,'ratio':n0/n1,'alpha':alpha,'dependence':dependence,
                         'rep':rep,'band':kind,'rate_coverage':float(np.all((a>=al-1e-12)&(a<=au+1e-12)&(b>=bl-1e-12)&(b<=bu+1e-12))),
                         'metric_coverage':float(np.all((truth>=rec['lower']-1e-12)&(truth<=rec['upper']+1e-12))),
                         'regret_coverage':float(np.all(actual<=rec['regret_bound']+1e-12)),
                         'strict_cert_fraction':float(rec['certified'].mean()),'learner_cert_fraction':float(rec['learner_certified'].mean()),
                         'strict_cert_any':float(rec['certified'].any()),'learner_cert_any':float(rec['learner_certified'].any()),
                         'mean_actual_regret':float(actual.mean()),'max_actual_regret':float(actual.max()),
                         'mean_regret_bound':float(rec['regret_bound'].mean()),'max_regret_bound':float(rec['regret_bound'].max())})
    return rows

def summarize(df,keys):
    g=df.groupby(keys,dropna=False);out=g.agg(repetitions=('rep','size'),rate_coverage=('rate_coverage','mean'),metric_coverage=('metric_coverage','mean'),regret_coverage=('regret_coverage','mean'),strict_cert_fraction=('strict_cert_fraction','mean'),learner_cert_fraction=('learner_cert_fraction','mean'),strict_cert_any=('strict_cert_any','mean'),learner_cert_any=('learner_cert_any','mean'),mean_actual_regret=('mean_actual_regret','mean'),max_actual_regret=('max_actual_regret','mean'),mean_regret_bound=('mean_regret_bound','mean'),max_regret_bound=('max_regret_bound','mean')).reset_index()
    for c in [x for x in out.columns if x in {'rate_coverage','metric_coverage','regret_coverage','strict_cert_fraction','learner_cert_fraction','strict_cert_any','learner_cert_any','mean_actual_regret','max_actual_regret','mean_regret_bound','max_regret_bound'}]:
        out[c+'_mcse']=g[c].std(ddof=1).fillna(0).to_numpy()/np.sqrt(out.repetitions)
    return out

def main():
    tic=time.perf_counter();allrows=[];seed0=2026100801
    # Broad sample-size / imbalance battery: 4 DGP x 7 n1 x 4 n0 ratios x 2 bands.
    for si,(name,dgp) in enumerate(DGPS.items()):
        for ni,n1 in enumerate([10,25,50,100,200,400,800]):
            for ri,ratio in enumerate([1,2,4,8]):
                rows=one_cell(name,dgp,n1,int(n1*ratio),.05,.7,300,seed0+si*100000+ni*1000+ri*17)
                allrows.extend(rows);print('broad',name,n1,ratio,flush=True)
    broad=pd.DataFrame(allrows);broad.to_csv(OUT/'broad_repetitions.csv',index=False);summarize(broad,['scenario','n1','ratio','band']).to_csv(OUT/'broad_summary.csv',index=False)
    # Alpha sensitivity at representative sample sizes.
    alpha_rows=[]
    for si,(name,dgp) in enumerate(DGPS.items()):
        for ai,alpha in enumerate([.01,.05,.1]):
            alpha_rows.extend(one_cell(name,dgp,100,400,alpha,.7,250,seed0+500000+si*10000+ai*100))
    alpha=pd.DataFrame(alpha_rows);alpha.to_csv(OUT/'alpha_repetitions.csv',index=False);summarize(alpha,['scenario','alpha','band']).to_csv(OUT/'alpha_summary.csv',index=False)
    # Dependence sensitivity for the two nontrivial settings.
    dep_rows=[]
    for si,name in enumerate(['separated','crossing']):
        for di,dep in enumerate([0,.3,.7,.9]):
            dep_rows.extend(one_cell(name,DGPS[name],100,400,.05,dep,300,seed0+700000+si*10000+di*100))
    dep=pd.DataFrame(dep_rows);dep.to_csv(OUT/'dependence_repetitions.csv',index=False);summarize(dep,['scenario','dependence','band']).to_csv(OUT/'dependence_summary.csv',index=False)
    # Grid-resolution sensitivity for the continuum correction, using the same
    # sampled calibration observations at every grid size.
    grid=[]
    for si,name in enumerate(['separated','crossing']):
        dgp=DGPS[name];a=population_rates(dgp['pos'],TS);b=population_rates(dgp['neg'],TS);rng=np.random.default_rng(seed0+900000+si)
        for rep in range(200):
            pos=sample_scores_dep(rng,100,dgp['pos'],False,False,.7);neg=sample_scores_dep(rng,400,dgp['neg'],False,False,.7)
            ah=np.concatenate([(pos[:,m,None]>=TS).mean(axis=0) for m in range(3)]);bh=np.concatenate([(neg[:,m,None]>=TS).mean(axis=0) for m in range(3)])
            band=bounds(ah,bh,100,400,alpha=.05,kind='cp',n_models=3,constants=CONSTANTS);sel=int(recommendations(ah,bh,band,PS,GROUPS)['selected'][12])
            for points in [25,51,101,201,401,801,1601]:
                q=interval_regret_bound(band,sel,.02,.5,points=points);grid.append({'scenario':name,'rep':rep,'grid_points':points,**q})
    pd.DataFrame(grid).to_csv(OUT/'grid_resolution.csv',index=False);summarize_grid=pd.DataFrame(grid).groupby(['scenario','grid_points']).agg(repetitions=('rep','size'),mean_bound=('bound','mean'),mean_grid_max=('grid_max','mean'),mean_correction=('lipschitz_correction','mean')).reset_index();summarize_grid.to_csv(OUT/'grid_resolution_summary.csv',index=False)
    # Population phase-complexity inventory.
    phase=[]
    for name,dgp in DGPS.items():
        a=population_rates(dgp['pos'],TS,dgp.get('atoms',False));b=population_rates(dgp['neg'],TS,dgp.get('atoms',False));ph=phase_diagram(a,b,lo=.02,hi=.5)
        phase.append({'scenario':name,'policy_count':len(a),'segments':len(ph['segments']),'pairwise_cut_count':ph['pairwise_cut_count'],'boundaries':len(ph['boundaries'])})
    pd.DataFrame(phase).to_csv(OUT/'population_phase_complexity.csv',index=False)
    meta={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'broad_repetitions_per_cell':300,'alpha_repetitions_per_cell':250,'dependence_repetitions_per_cell':300,'grid_repetitions_per_scenario':200,'n1':[10,25,50,100,200,400,800],'n0_ratio':[1,2,4,8],'alpha':[.01,.05,.1],'dependence':[0,.3,.7,.9],'bands':['cp','dkw'],'prevalence_grid_points':25,'seed':seed0,'scope':'Additive sensitivity battery; all results are fixed-DGP Monte Carlo summaries, not population inference.'}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('DONE',len(broad),len(alpha),len(dep),len(grid),'rows',time.perf_counter()-tic,'seconds')
if __name__=='__main__':main()
