from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from policy_certificate import bounds,recommendations,macro_f1
from run_development import ARCHIVE,METHODS,POLICIES,SEEDS
from extended_experiments import sample_scores_dep
from scipy.special import betainc
R=Path(__file__).resolve().parent;O=R/'results/extended';O.mkdir(exist_ok=True)

def pool_ablation():
    rows=[]
    for grid_index,nt in enumerate([6,12,22]):
        ts=np.r_[np.linspace(0,1,nt-1),np.inf]
        aa=np.concatenate([1-betainc(a,b,np.clip(ts,0,1)) for a,b in [(8,2),(3,3),(2,4)]])
        bb=np.concatenate([1-betainc(a,b,np.clip(ts,0,1)) for a,b in [(2,8),(3,3),(4,2)]])
        cons=np.tile(np.r_[1,np.zeros(nt-2),-1],3)
        # All-negative policies at both t=1 and infinity are structurally
        # identical almost surely for these continuous beta distributions.
        cons[np.tile(np.arange(nt)==nt-2,3)]=-1
        unique=[i for i in range(len(aa)) if cons[i]==0]
        unique += [int(np.flatnonzero(cons==v)[0]) for v in [-1,1]]
        for n1 in [25,100,400]:
            rng=np.random.default_rng(2026100810+grid_index*10000+n1)
            for rep in range(500):
                pos=sample_scores_dep(rng,n1,[(8,2),(3,3),(2,4)],dependence=.7)
                neg=sample_scores_dep(rng,4*n1,[(2,8),(3,3),(4,2)],dependence=.7)
                ah=np.concatenate([(pos[:,m,None]>=ts).mean(0) for m in range(3)])
                bh=np.concatenate([(neg[:,m,None]>=ts).mean(0) for m in range(3)])
                ps=np.linspace(.02,.5,25)
                for variant,ix in [('labelled',np.arange(len(aa))),('structural_unique',np.array(unique))]:
                    a,b=aa[ix],bb[ix];truth=macro_f1(a[:,None],b[:,None],ps[None,:]);oracle=truth.max(0)
                    for kind in ['cp','dkw']:
                        band=bounds(ah[ix],bh[ix],n1,4*n1,kind=kind,n_models=3,constants=cons[ix]);rec=recommendations(ah[ix],bh[ix],band,ps)
                        regret=oracle-truth[rec['selected'],np.arange(25)]
                        rows.append(dict(thresholds_per_score=nt,n1=n1,rep=rep,variant=variant,policy_count=len(ix),band=kind,
                                         strict_cert_fraction=rec['certified'].mean(),strict_cert_any=rec['certified'].any(),
                                         false_strict_certificate=bool(np.any(rec['certified']&(regret>1e-10))),mean_regret=regret.mean(),mean_bound=rec['regret_bound'].mean()))
            print('pool',nt,n1,flush=True)
    df=pd.DataFrame(rows);df.to_csv(O/'pool_ablation_repetitions.csv',index=False)
    keys=['thresholds_per_score','n1','variant','policy_count','band'];g=df.groupby(keys)
    out=g.agg(repetitions=('rep','size'),strict_cert_fraction=('strict_cert_fraction','mean'),strict_cert_any=('strict_cert_any','mean'),false_strict_certificate=('false_strict_certificate','mean'),mean_regret=('mean_regret','mean'),mean_bound=('mean_bound','mean')).reset_index()
    for c in ['strict_cert_fraction','mean_regret','mean_bound']:out[c+'_mcse']=g[c].std().to_numpy()/np.sqrt(out.repetitions)
    out.to_csv(O/'pool_ablation_summary.csv',index=False)

def archive_resplit():
    configs=json.loads((ARCHIVE/'openml_source_schema_audit.json').read_text(encoding='utf-8'))['datasets']
    prior={c['name']:c['positive_n']/c['n'] for c in configs};rows=[]
    blocks=sorted((ARCHIVE/'results').glob('*/outer_predictions.parquet'));assert len(blocks)==135
    for bi,path in enumerate(blocks):
        pred=pd.read_parquet(path);meta=json.loads((path.parent/'complete.json').read_text());dataset=meta['dataset'];split_seed=int(meta['split_seed']);fold=int(meta['fold'])
        base=pred[pred.method=='unweighted_lr'].sort_values('row_index');ids=base.row_index.to_numpy();y=base.y_true.to_numpy();p=prior[dataset];ps=np.array([p/2,p,min(2*p,.8)])
        metrics=pd.read_csv(path.parent/'metrics.csv',dtype={'model_seed':str});pools={}
        for seed in SEEDS:
            pool=[]
            for method in METHODS:
                ss='deterministic' if method.endswith('_lr') else str(seed)
                f=pred[(pred.method==method)&(pred.model_seed.astype(str)==ss)].sort_values('row_index');assert np.array_equal(f.row_index,ids)
                mm=metrics[(metrics.method==method)&(metrics.model_seed==ss)].set_index('policy')
                for policy in POLICIES:
                    pool.append(f.default_prediction.to_numpy() if policy=='default' else f.positive_probability.to_numpy()>=float(mm.loc[policy,'threshold']))
            pools[seed]=np.array(pool,dtype=np.uint8)
        for rep in range(10):
            rng=np.random.default_rng(2026100820+10000*split_seed+100*fold+rep)
            permutations=[rng.permutation(np.flatnonzero(y==c)) for c in [0,1]]
            for frac in [.25,.5,.75]:
                cuts=[int(len(v)*frac) for v in permutations];assert min(cuts)>0
                cal=np.concatenate([v[:cut] for v,cut in zip(permutations,cuts)]);eva=np.concatenate([v[cut:] for v,cut in zip(permutations,cuts)])
                assert not np.intersect1d(ids[cal],ids[eva]).size
                n1=int(y[cal].sum());n0=len(cal)-n1
                for seed,pool in pools.items():
                    ah=pool[:,cal[y[cal]==1]].mean(1);bh=pool[:,cal[y[cal]==0]].mean(1)
                    ae=pool[:,eva[y[eva]==1]].mean(1);be=pool[:,eva[y[eva]==0]].mean(1)
                    truth=macro_f1(ae[:,None],be[:,None],ps[None,:]);oracle=truth.max(0)
                    for kind in ['cp','dkw']:
                        band=bounds(ah,bh,n1,n0,kind=kind,n_models=8);rec=recommendations(ah,bh,band,ps,np.repeat(np.arange(4),3))
                        j=1;s=int(rec['selected'][j]);e=int(rec['empirical'][j])
                        rows.append(dict(dataset=dataset,split_seed=split_seed,fold=fold,forest_seed=seed,resplit=rep,calibration_fraction=frac,n1=n1,n0=n0,band=kind,
                                         strict_any_three=bool(rec['certified'].any()),learner_any_three=bool(rec['learner_certified'].any()),
                                         source_strict=bool(rec['certified'][1]),source_learner=bool(rec['learner_certified'][1]),
                                         source_bound=rec['regret_bound'][1],source_lcb_regret=oracle[j]-truth[s,j],source_empirical_regret=oracle[j]-truth[e,j],selection=s,empirical_selection=e))
        if bi%15==14:
            print('archive resplit',bi+1,'/135',flush=True)
    df=pd.DataFrame(rows);df.to_csv(O/'archive_resplit.csv',index=False)
    for view,sub in [('primary',df[df.forest_seed==101]),('all_views',df)]:
        out=sub.groupby(['dataset','calibration_fraction','band']).agg(evaluations=('resplit','size'),min_n1=('n1','min'),median_n1=('n1','median'),source_strict_fraction=('source_strict','mean'),source_learner_fraction=('source_learner','mean'),any_three_learner_fraction=('learner_any_three','mean'),mean_bound=('source_bound','mean'),mean_lcb_regret=('source_lcb_regret','mean'),mean_empirical_regret=('source_empirical_regret','mean')).reset_index()
        out.to_csv(O/f'archive_resplit_{view}_summary.csv',index=False)
    print('archive rows',len(df),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['pool','archive-resplit'])
    args=parser.parse_args()
    {'pool':pool_ablation,'archive-resplit':archive_resplit}[args.mode]()
