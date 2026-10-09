from __future__ import annotations
import argparse,json,time,platform,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betaincinv,betainc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score,f1_score,balanced_accuracy_score
from policy_certificate import (macro_f1,choose_threshold,rates,bounds,
    recommendations,phase_diagram,frontier_phase_diagram,interval_regret_bound)

HERE=Path(__file__).resolve().parent
ARCHIVE=Path(os.environ.get('CISSC_ARCHIVE_ROOT', str(HERE/'archive_not_included')))
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
METHODS=['unweighted_lr','weighted_lr','rf','balanced_rf']
POLICIES=['default','oof_macro_f1','oof_balanced_accuracy']
SEEDS=[101,202,303,404,505]
DGPS={
 'identical':{'pos':[(3,2)]*3,'neg':[(2,4)]*3,'shared':True},
 'separated':{'pos':[(8,2),(3,3),(2,4)],'neg':[(2,8),(3,3),(4,2)]},
 'crossing':{'pos':[(2,1),(8,4),(1.2,1)],'neg':[(1,1)]*3},
 'atoms':{'pos':[(2,1),(8,4),(1.2,1)],'neg':[(1,1)]*3,'atoms':True},
}

def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def sample_scores(rng,n,param,shared=False,atoms=False):
    u=rng.uniform(size=(n,1));
    if not shared:
        independent=rng.uniform(size=(n,3));use_shared=rng.uniform(size=(n,3))<.7
        u=np.where(use_shared,u,independent)
    x=np.column_stack([betaincinv(a,b,u[:,0 if shared else m]) for m,(a,b) in enumerate(param)])
    return np.floor(4*x+.5)/4 if atoms else x

def population_rates(param,thresholds,atoms=False):
    t=thresholds.copy()
    if atoms:t=(np.ceil(4*t)-.5)/4
    t=np.clip(t,0,1)
    return np.concatenate([1-betainc(a,b,t) for a,b in param])

def sup_cdf_errors(x,param,atoms=False):
    """Exact sup empirical/population CDF gap for the known simulation DGP.

    For continuous CDFs the extrema occur immediately before/at observations;
    for the discrete DGP include every support atom, including unobserved ones.
    """
    out=[]
    for m,(a,b) in enumerate(param):
        sx=np.sort(x[:,m]);n=len(sx)
        if atoms:
            support=np.linspace(0,1,5)
            er=np.searchsorted(sx,support,side='right')/n
            el=np.searchsorted(sx,support,side='left')/n
            fr=betainc(a,b,np.clip(support+.125,0,1))
            fl=betainc(a,b,np.clip(support-.125,0,1))
            out.append(max(np.max(abs(er-fr)),np.max(abs(el-fl))))
        else:
            f=betainc(a,b,sx);out.append(max(np.max(abs(np.arange(1,n+1)/n-f)),np.max(abs(np.arange(n)/n-f))))
    return np.array(out)

def simulate(reps=1000):
    tic=time.perf_counter();ts=np.r_[np.linspace(0,1,11),np.inf];ps=np.linspace(.02,.5,25)
    grid=np.tile(ts,3);constants=np.tile(np.r_[1,np.zeros(10),-1],3)
    all_rows=[];parameters={}
    for scenario_index,(name,dgp) in enumerate(DGPS.items()):
        a=population_rates(dgp['pos'],ts,dgp.get('atoms',False));b=population_rates(dgp['neg'],ts,dgp.get('atoms',False))
        truth=macro_f1(a[:,None],b[:,None],ps[None,:]);oracle=truth.max(axis=0)
        parameters[name]={**dgp,'population_tpr':a.tolist(),'population_fpr':b.tolist(),
                          'population_phase':phase_diagram(a,b,lo=.02,hi=.5)}
        for size_index,n1 in enumerate([25,100,400]):
            n0=4*n1;rng=np.random.default_rng(20261008+10000*scenario_index+100*size_index)
            for r in range(reps):
                pos=sample_scores(rng,n1,dgp['pos'],dgp.get('shared',False),dgp.get('atoms',False))
                neg=sample_scores(rng,n0,dgp['neg'],dgp.get('shared',False),dgp.get('atoms',False))
                full_dkw_covered=bool(np.all(sup_cdf_errors(pos,dgp['pos'],dgp.get('atoms',False))<=np.sqrt(np.log(12/.05)/(2*n1))) and
                    np.all(sup_cdf_errors(neg,dgp['neg'],dgp.get('atoms',False))<=np.sqrt(np.log(12/.05)/(2*n0))))
                ah=np.concatenate([(pos[:,m,None]>=ts).mean(axis=0) for m in range(3)])
                bh=np.concatenate([(neg[:,m,None]>=ts).mean(axis=0) for m in range(3)])
                emp=macro_f1(ah[:,None],bh[:,None],ps[None,:]);ep=emp.argmax(axis=0)
                nominal=int(macro_f1(ah,bh,.10).argmax())
                minimax=int((emp.max(axis=0)-emp).max(axis=1).argmin())
                for kind in ['dkw','cp']:
                    band=bounds(ah,bh,n1,n0,kind=kind,n_models=3,constants=constants)
                    groups=np.repeat(np.arange(3),len(ts))
                    rec=recommendations(ah,bh,band,ps,groups);sel=rec['selected'];cols=np.arange(len(ps))
                    actual=oracle-truth[sel,cols]
                    al,au,bl,bu=band
                    rate_covered=bool(np.all((a>=al-1e-12)&(a<=au+1e-12)&(b>=bl-1e-12)&(b<=bu+1e-12)))
                    metric_covered=bool(np.all((truth>=rec['lower']-1e-12)&(truth<=rec['upper']+1e-12)))
                    false=bool(np.any(rec['certified']&(actual>1e-10)))
                    selected_group_oracle=np.array([truth[groups==groups[sel[j]],j].max() for j in cols])
                    row={'scenario':name,'n1':n1,'n0':n0,'repetition':r,'band':kind,
                      'rate_covered':rate_covered,'metric_covered':metric_covered,'false_certification_any':false,
                      'dkw_full_score_band_covered':full_dkw_covered if kind=='dkw' else None,
                      'certification_any':bool(rec['certified'].any()),'certification_grid_fraction':rec['certified'].mean(),
                      'learner_certification_grid_fraction':rec['learner_certified'].mean(),
                      'learner_certification_any':bool(rec['learner_certified'].any()),
                      'false_learner_certification_any':bool(np.any(rec['learner_certified']&(oracle-selected_group_oracle>1e-10))),
                      'regret_bound_covered':bool(np.all(actual<=rec['regret_bound']+1e-12)),
                      'mean_lcb_regret':actual.mean(),'worst_lcb_regret':actual.max(),
                      'mean_empirical_regret':(oracle-truth[ep,cols]).mean(),
                      'worst_nominal_regret':float((oracle-truth[nominal]).max()),
                      'worst_minimax_regret':float((oracle-truth[minimax]).max()),
                      'mean_regret_bound':rec['regret_bound'].mean(),'mean_selected_utility':truth[sel,cols].mean()}
                    all_rows.append(row)
            print(f'simulation {name} n1={n1} reps={reps}',flush=True)
    df=pd.DataFrame(all_rows);df.to_csv(OUT/'simulation_repetitions.csv',index=False)
    measures=[c for c in df if c not in ['scenario','n1','n0','repetition','band']]
    summaries=[]
    for key,g in df.groupby(['scenario','n1','band']):
        row=dict(zip(['scenario','n1','band'],key));row['repetitions']=len(g)
        for c in measures:
            row[c]=float(g[c].mean());row[c+'_mcse']=float(g[c].std(ddof=1)/np.sqrt(len(g)))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT/'simulation_summary.csv',index=False)
    save('simulation_metadata.json',{'seed':20261008,'repetitions_per_cell':reps,'n1':[25,100,400],
        'n0_ratio':4,'thresholds':[float(x) if np.isfinite(x) else 'infinity' for x in ts],
        'prevalences':ps.tolist(),'alpha':.05,'dgp':parameters,'seconds':time.perf_counter()-tic,
        'python':platform.python_version()})

def application():
    configs=json.loads((ARCHIVE/'openml_source_schema_audit.json').read_text(encoding='utf-8'))['datasets']
    prior={c['name']:c['positive_n']/c['n'] for c in configs}
    rows=[];phases=[];counts=[];rate_rows=[];tic=time.perf_counter()
    for ip,path in enumerate(sorted((ARCHIVE/'results').glob('*/outer_predictions.parquet'))):
        block=path.parent;pred=pd.read_parquet(path);metrics=pd.read_csv(block/'metrics.csv',dtype={'model_seed':str},float_precision='round_trip')
        name=pred.dataset.iloc[0];s=int(pred.split_seed.iloc[0]);fold=int(pred.fold.iloc[0]);p=prior[name]
        base=pred[(pred.method=='unweighted_lr')].sort_values('row_index');ids=base.row_index.to_numpy();y=base.y_true.to_numpy()
        rng=np.random.default_rng(20261008+100*s+fold);cal=[];eva=[]
        for cls in [0,1]:
            ix=np.flatnonzero(y==cls);rng.shuffle(ix);cut=len(ix)//2;cal.extend(ix[:cut]);eva.extend(ix[cut:])
        cal=np.sort(cal);eva=np.sort(eva);assert not np.intersect1d(ids[cal],ids[eva]).size
        n1=int(y[cal].sum());n0=len(cal)-n1;counts.append({'dataset':name,'split_seed':s,'fold':fold,'calibration_n1':n1,'calibration_n0':n0,'evaluation_n1':int(y[eva].sum()),'evaluation_n0':len(eva)-int(y[eva].sum())})
        # All 12 models are split on the same row IDs before any policy selection.
        for seed in SEEDS:
            ah=[];bh=[];ae=[];be=[];labels=[];ts=[];scores=[]
            for method in METHODS:
                ss='deterministic' if method.endswith('_lr') else str(seed)
                g=pred[(pred.method==method)&(pred.model_seed.astype(str)==ss)].sort_values('row_index');assert np.array_equal(g.row_index,ids)
                mm=metrics[(metrics.method==method)&(metrics.model_seed==ss)].set_index('policy')
                for policy in POLICIES:
                    t=float(mm.loc[policy,'threshold']);prob=g.positive_probability.to_numpy()
                    # LR/RF class prediction uses strict .5 at exact ties. Retain
                    # stored default predictions rather than assuming >= .5.
                    cpred=g.default_prediction.to_numpy() if policy=='default' else (prob>=t)
                    ah.append(cpred[cal][y[cal]==1].mean());bh.append(cpred[cal][y[cal]==0].mean())
                    ae.append(cpred[eva][y[eva]==1].mean());be.append(cpred[eva][y[eva]==0].mean())
                    labels.append(method+':'+policy);ts.append(t)
            ah,bh,ae,be=map(np.array,[ah,bh,ae,be]);lo=p/2;hi=min(2*p,.8);ps=np.array([lo,p,hi])
            for k,label in enumerate(labels):rate_rows.append({'dataset':name,'split_seed':s,'fold':fold,'forest_seed':seed,
                'policy_label':label,'threshold':ts[k],'calibration_tpr':ah[k],'calibration_fpr':bh[k],
                'evaluation_tpr':ae[k],'evaluation_fpr':be[k]})
            phase=frontier_phase_diagram(ae,be,labels,lo=lo,hi=hi)
            reference=phase_diagram(ae,be,labels,lo=lo,hi=hi)
            assert len(phase['segments'])==len(reference['segments'])
            assert all(abs(a['right']-b['right'])<1e-8 and set(a['winners'])==set(b['winners']) for a,b in zip(phase['segments'],reference['segments']))
            phases.append({'dataset':name,'split_seed':s,'fold':fold,'forest_seed':seed,'source':'evaluation conditional rates; descriptive, not certified',**phase})
            truth=macro_f1(ae[:,None],be[:,None],ps[None,:]);oracle=truth.max(axis=0)
            for kind in ['dkw','cp']:
                # Default est.predict need not be a uniform >=.5 rule at
                # numerical ties. Treat its hard prediction as another fixed
                # {0,1}-valued score: 4 probability + 4 hard-score functions.
                band=bounds(ah,bh,n1,n0,kind=kind,n_models=8)
                rec=recommendations(ah,bh,band,ps,np.repeat(np.arange(4),3))
                nominal=int(rec['selected'][1]);whole=interval_regret_bound(band,nominal,lo,hi)
                for j,pi in enumerate(ps):
                    sel=int(rec['selected'][j]);emp=int(rec['empirical'][j]);actual=oracle[j]-truth[sel,j]
                    rows.append({'dataset':name,'split_seed':s,'fold':fold,'forest_seed':seed,'band':kind,
                     'prevalence_position':['low','source','high'][j],'prevalence':float(pi),'calibration_n1':n1,'calibration_n0':n0,
                     'lcb_selection':labels[sel],'empirical_selection':labels[emp],'evaluation_winner':labels[int(truth[:,j].argmax())],
                     'strict_certified':bool(rec['certified'][j]),'learner_certified':bool(rec['learner_certified'][j]),
                     'regret_bound':float(rec['regret_bound'][j]),
                     'evaluation_lcb_utility':float(truth[sel,j]),'evaluation_empirical_utility':float(truth[emp,j]),
                     'evaluation_lcb_regret':float(actual),'evaluation_empirical_regret':float(oracle[j]-truth[emp,j]),
                     'nominal_policy_interval_regret_bound':whole['bound'],'interval_lipschitz_correction':whole['lipschitz_correction'],
                     'evaluation_rate_bound_violations_descriptive':int(np.sum((ae<band[0])|(ae>band[1])|(be<band[2])|(be>band[3])))})
        if ip%15==14:print(f'application {ip+1}/135',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'archive_application.csv',index=False)
    pd.DataFrame(rate_rows).to_csv(OUT/'archive_conditional_rates.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'archive_calibration_counts.csv',index=False)
    save('archive_phases.json',phases)
    df=pd.DataFrame(rows);summary=df[df.forest_seed==101].groupby(['dataset','band','prevalence_position']).agg(
        blocks=('fold','size'),certification_fraction=('strict_certified','mean'),learner_certification_fraction=('learner_certified','mean'),mean_bound=('regret_bound','mean'),
        mean_lcb_utility=('evaluation_lcb_utility','mean'),mean_empirical_utility=('evaluation_empirical_utility','mean'),
        mean_lcb_regret=('evaluation_lcb_regret','mean'),mean_empirical_regret=('evaluation_empirical_regret','mean')).reset_index()
    summary.to_csv(OUT/'archive_primary_summary.csv',index=False)
    save('application_metadata.json',{'blocks':135,'forest_seed_views':SEEDS,'primary_forest_seed':101,
        'policies_per_view':12,'split_seed':20261008,'calibration_fraction_per_class':'floor(nc/2)/nc',
        'source_prevalence':prior,'seconds':time.perf_counter()-tic,
        'scope':'Descriptive sensitivity on fixed tasks; no real deployment data; no IID/general-population guarantee asserted.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['simulation','application']);p.add_argument('--reps',type=int,default=1000);args=p.parse_args()
    {'simulation':lambda:simulate(args.reps),'application':application}[args.mode]()
