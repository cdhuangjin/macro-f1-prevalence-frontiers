"""Validate frozen derived results without rerunning simulations or training."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from policy_certificate import macro_f1,bounds,recommendations,interval_regret_bound

R=Path(__file__).resolve().parent/'results'

def read(name):
    return pd.read_csv(R/name,float_precision='round_trip')

def close(actual,expected,label):
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12,equal_nan=True,err_msg=label)

def compare(expected,stored,keys,label):
    assert not expected.duplicated(keys).any() and not stored.duplicated(keys).any(),label
    expected=expected.set_index(keys).sort_index();stored=stored.set_index(keys).sort_index()
    assert expected.index.equals(stored.index),label+' group keys'
    assert set(expected.columns)==set(stored.columns),label+' columns'
    for col in expected.columns:
        close(expected[col].to_numpy(dtype=float),stored[col].to_numpy(dtype=float),label+' '+col)
    print('Verified',label,flush=True)

def summary(raw_name,summary_name,keys,rep='rep',mcse_fill=False):
    raw=read(raw_name);stored=read(summary_name);g=raw.groupby(keys,dropna=False)
    expected=g[rep].size().rename('repetitions').reset_index()
    for col in stored.columns:
        if col in keys or col=='repetitions':continue
        if col.endswith('_mcse'):
            base=col[:-5];std=g[base].agg(lambda s:s.astype(float).std(ddof=1))
            if mcse_fill:std=std.fillna(0)
            values=std.to_numpy()/np.sqrt(expected.repetitions)
        else:values=g[col].agg(lambda s:s.astype(float).mean()).to_numpy()
        expected[col]=values
    compare(expected,stored,keys,summary_name)
    return raw

def mapped_summary(raw,filename,keys,aggregations):
    compare(raw.groupby(keys).agg(**aggregations).reset_index(),read(filename),keys,filename)

def main():
    manifest=json.loads((R/'SHA256.json').read_text())
    actual={p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file() and p.name!='SHA256.json'}
    assert actual==set(manifest),'Result inventory changed'
    for name,digest in manifest.items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name+' checksum'
    print('Verified result checksums',len(manifest),flush=True)
    original=summary('simulation_repetitions.csv','simulation_summary.csv',['scenario','n1','band'],rep='repetition')
    broad=summary('extended/broad_repetitions.csv','extended/broad_summary.csv',['scenario','n1','ratio','band'],mcse_fill=True)
    alpha=summary('extended/alpha_repetitions.csv','extended/alpha_summary.csv',['scenario','alpha','band'],mcse_fill=True)
    dep=summary('extended/dependence_repetitions.csv','extended/dependence_summary.csv',['scenario','dependence','band'],mcse_fill=True)
    pool=summary('extended/pool_ablation_repetitions.csv','extended/pool_ablation_summary.csv',['thresholds_per_score','n1','variant','policy_count','band'])
    orth=summary('extended/orthogonal_pool_ablation_repetitions.csv','extended/orthogonal_pool_ablation_summary.csv',['frontier_density','total_k','n1','band'],mcse_fill=True)
    grid=read('extended/grid_resolution.csv')
    mapped_summary(grid,'extended/grid_resolution_summary.csv',['scenario','grid_points'],dict(repetitions=('rep','size'),mean_bound=('bound','mean'),mean_grid_max=('grid_max','mean'),mean_correction=('lipschitz_correction','mean')))
    close(grid.bound,np.minimum(1,grid.grid_max+grid.lipschitz_correction),'grid correction identity')
    sizes=[original[['scenario','n1','repetition']].drop_duplicates().shape[0],broad[['scenario','n1','ratio','rep']].drop_duplicates().shape[0],alpha[['scenario','alpha','rep']].drop_duplicates().shape[0],dep[['scenario','dependence','rep']].drop_duplicates().shape[0],grid[['scenario','rep']].drop_duplicates().shape[0],pool[['thresholds_per_score','n1','rep']].drop_duplicates().shape[0],orth[['frontier_density','total_k','n1','rep']].drop_duplicates().shape[0]]
    assert sizes==[12000,33600,3000,2400,400,4500,16000] and sum(sizes)==71900,sizes
    print('Verified 71,900 calibration draws (paired bands/grid reuse counted once)',flush=True)
    app=read('archive_application.csv');primary=app[app.forest_seed==101]
    assert primary.dataset.nunique()==9
    assert len(primary[['dataset','split_seed','fold']].drop_duplicates())==135
    assert not app.strict_certified.any() and not app.learner_certified.any()
    mapped_summary(primary,'archive_primary_summary.csv',['dataset','band','prevalence_position'],dict(blocks=('fold','size'),certification_fraction=('strict_certified','mean'),learner_certification_fraction=('learner_certified','mean'),mean_bound=('regret_bound','mean'),mean_lcb_utility=('evaluation_lcb_utility','mean'),mean_empirical_utility=('evaluation_empirical_utility','mean'),mean_lcb_regret=('evaluation_lcb_regret','mean'),mean_empirical_regret=('evaluation_empirical_regret','mean')))
    phases=json.loads((R/'archive_phases.json').read_text())
    assert len(phases)==675 and sum(len(p['segments'])>1 for p in phases if p['forest_seed']==101)==102
    counts=read('archive_calibration_counts.csv').set_index(['dataset','split_seed','fold'])
    rates=read('archive_conditional_rates.csv');lookup=app.set_index(['dataset','split_seed','fold','forest_seed','band','prevalence_position']).sort_index()
    for key,block in rates.groupby(['dataset','split_seed','fold','forest_seed'],sort=False):
        labels=block.policy_label.to_list();assert len(labels)==12
        ah,bh,ae,be=[block[c].to_numpy() for c in ['calibration_tpr','calibration_fpr','evaluation_tpr','evaluation_fpr']]
        count=counts.loc[key[:3]]
        for kind in ['cp','dkw']:
            records=lookup.loc[(*key,kind)].reindex(['low','source','high']);ps=records.prevalence.to_numpy()
            band=bounds(ah,bh,int(count.calibration_n1),int(count.calibration_n0),kind=kind,n_models=8)
            rec=recommendations(ah,bh,band,ps,np.repeat(np.arange(4),3))
            truth=macro_f1(ae[:,None],be[:,None],ps[None,:]);col=np.arange(3)
            close(rec['regret_bound'],records.regret_bound,'archive regret bound')
            close(truth[rec['selected'],col],records.evaluation_lcb_utility,'archive LCB utility')
            close(truth[rec['empirical'],col],records.evaluation_empirical_utility,'archive empirical utility')
            close(truth.max(0)-truth[rec['selected'],col],records.evaluation_lcb_regret,'archive evaluation regret')
            assert list(np.array(labels)[rec['selected']])==records.lcb_selection.to_list()
            assert np.array_equal(rec['certified'],records.strict_certified)
            assert np.array_equal(rec['learner_certified'],records.learner_certified)
            whole=interval_regret_bound(band,int(rec['selected'][1]),ps[0],ps[2])
            close(records.nominal_policy_interval_regret_bound,whole['bound'],'archive interval bound')
            close(records.interval_lipschitz_correction,whole['lipschitz_correction'],'archive correction')
    print('Recomputed 4,050 archived application records from public aggregate rates',flush=True)
    resplit=read('extended/archive_resplit.csv')
    agg=dict(evaluations=('resplit','size'),min_n1=('n1','min'),median_n1=('n1','median'),source_strict_fraction=('source_strict','mean'),source_learner_fraction=('source_learner','mean'),any_three_learner_fraction=('learner_any_three','mean'),mean_bound=('source_bound','mean'),mean_lcb_regret=('source_lcb_regret','mean'),mean_empirical_regret=('source_empirical_regret','mean'))
    for view,sub in [('primary',resplit[resplit.forest_seed==101]),('all_views',resplit)]:
        mapped_summary(sub,f'extended/archive_resplit_{view}_summary.csv',['dataset','calibration_fraction','band'],agg)
    diag=read('diagnostics/archive_zero_certificate_diagnostics.csv')
    close(diag.strict_margin,diag.selected_point_gap-diag.selected_lower_loss-diag.competitor_upper_inflation,'strict margin decomposition')
    close(diag.learner_margin,diag.learner_point_gap-diag.learner_penalty,'learner margin decomposition')
    assert not diag.strict_certified.any() and not diag.learner_certified.any()
    diag=diag[(diag.forest_seed==101)&(diag.position=='source')]
    keys=['dataset','band'];g=diag.groupby(keys)
    stored=read('diagnostics/archive_zero_certificate_primary_source_summary.csv');expected=g.size().rename('records').reset_index()
    for col in stored.columns:
        if col in keys or col=='records':continue
        base,stat=col.rsplit('_',1);expected[col]=g[base].agg(stat).to_numpy()
    compare(expected,stored,keys,'primary zero-certificate diagnostics')
    med=read('diagnostics/archive_zero_certificate_medians.csv')
    compare(g[list(med.columns.difference(keys))].median().reset_index(),med,keys,'diagnostic medians')
    runtime=read('runtime/runtime_repetitions.csv')
    mapped_summary(runtime,'runtime/runtime_summary.csv',['family','K','method'],dict(seconds_median=('seconds','median'),seconds_min=('seconds','min'),seconds_max=('seconds','max'),root_solves=('root_solves','first'),segments=('segments','first')))
    print('All stored-result checks passed. No training or simulation was rerun.',flush=True)

if __name__=='__main__':main()
