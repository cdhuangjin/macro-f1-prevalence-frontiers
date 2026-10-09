"""Deterministic complete-function runtime comparison; no calibration draws."""
import time,json,platform,hashlib
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;A=R/'results/runtime';A.mkdir(parents=True,exist_ok=True)
from policy_certificate import frontier_phase_diagram,phase_diagram,macro_f1
rows=[];checks=[]
for family in ['sqrt','diagonal']:
 for k in [32,64,128,256,1024]:
  b=np.linspace(.0001,.9999,k);a=np.sqrt(b) if family=='sqrt' else b.copy()
  for rep in range(3):
   t=time.perf_counter();fast=frontier_phase_diagram(a,b,lo=.01,hi=.99);sec=time.perf_counter()-t
   rows.append(dict(family=family,K=k,rep=rep,method='stack_complete',seconds=sec,root_solves=fast['root_solves'],segments=len(fast['segments']),frontier_size=fast['frontier_size']))
   if k<=256:
    t=time.perf_counter();slow=phase_diagram(a,b,lo=.01,hi=.99);sec=time.perf_counter()-t
    rows.append(dict(family=family,K=k,rep=rep,method='all_pairs_complete',seconds=sec,root_solves=k*(k-1)//2,segments=len(slow['segments']),frontier_size=k))
    assert len(fast['segments'])==len(slow['segments'])
    assert all(f['winners']==s['winners'] and abs(f['left']-s['left'])<1e-8 and abs(f['right']-s['right'])<1e-8 for f,s in zip(fast['segments'],slow['segments']))
   assert not fast['float_resolution_unresolved']
  for p in np.linspace(.01001,.98999,301):
   seg=next(s for s in fast['segments'] if s['left']<=p<=s['right']);values=macro_f1(a,b,p);assert values.max()-values[list(seg['winners'])].max()<1e-12
  pd.DataFrame(rows).to_csv(A/'runtime_repetitions.csv',index=False)
  checks.append(dict(family=family,K=k,matched_all_pairs=k<=256,grid_points=301))
  print('benchmark',family,k,'segments',len(fast['segments']),'roots',fast['root_solves'],flush=True)
raw=pd.DataFrame(rows);raw.to_csv(A/'runtime_repetitions.csv',index=False)
s=raw.groupby(['family','K','method']).agg(seconds_median=('seconds','median'),seconds_min=('seconds','min'),seconds_max=('seconds','max'),root_solves=('root_solves','first'),segments=('segments','first')).reset_index();s.to_csv(A/'runtime_summary.csv',index=False)
meta=dict(python=platform.python_version(),numpy=np.__version__,repetitions=3,interval=[.01,.99],source_sha256=hashlib.sha256((R/'policy_certificate.py').read_bytes()).hexdigest(),checks=checks,scope='Complete functions including boundary label scans. Shared pair-root helper means timing baseline is not independent root-accuracy proof. All-pairs omitted at K1024 by design; no speed extrapolation.')
(A/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(s.to_string(index=False))
