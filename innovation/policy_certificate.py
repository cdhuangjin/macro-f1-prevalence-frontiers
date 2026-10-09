"""Finite frozen-policy macro-F1 sensitivity and simultaneous rate certificates.

Positive prediction is score >= threshold. Population guarantee requires pure
label shift, frozen scores and independent class-conditional IID calibration.
CP additionally requires policies frozen independently of calibration labels.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.stats import beta
from scipy.optimize import brentq
import mpmath as mp


def macro_f1(a,b,p):
    a,b,p=np.asarray(a),np.asarray(b),np.asarray(p)
    if np.any((p<=0)|(p>=1)): raise ValueError('prevalence must be strictly between 0 and 1')
    return p*a/(p*(1+a)+(1-p)*b)+(1-p)*(1-b)/((1-p)*(2-b)+p*(1-a))


def curve_polynomials(a,b):
    e=P([b,1-a-b]); q=P([b,1+a-b]); d=q*(P([2])-q)
    return d-e,d


def real_roots(poly,lo,hi,tol=1e-9):
    co=poly.coef.copy(); scale=max(np.max(np.abs(co)),1e-300)
    while len(co)>1 and abs(co[-1])<1e-13*scale: co=co[:-1]
    if len(co)<=1: return []
    roots=P(co).roots()
    out=[]
    for r in roots:
        if abs(r.imag)<tol and lo+tol<r.real<hi-tol:
            v=float(r.real)
            if abs(P(co)(v))<1e-7*scale and not any(abs(v-x)<tol for x in out):out.append(v)
    return sorted(out)


def _mp_metric(a,b,p):
    a,b,p=mp.mpf(float(a)),mp.mpf(float(b)),mp.mpf(p)
    return p*a/(p*(1+a)+(1-p)*b)+(1-p)*(1-b)/((1-p)*(2-b)+p*(1-a))


def _pair_diff(a,b,c,d,p):
    # Preserve the actual binary float inputs; a small ROC difference is not
    # mathematical identity. High precision avoids cancellation of utilities.
    value=float(macro_f1(a,b,p)-macro_f1(c,d,p))
    if max(abs(a-c),abs(b-d))<1e-8 or abs(value)<1e-11 or min(p,1-p)<1e-8 or not np.isfinite(value):
        with mp.workdps(80):return float(_mp_metric(a,b,p)-_mp_metric(c,d,p))
    return value


def _mp_crossing(a,b,c,d,lo,hi):
    with mp.workdps(80):
        left,right=mp.mpf(float(lo)),mp.mpf(float(hi))
        def f(p):return _mp_metric(a,b,p)-_mp_metric(c,d,p)
        fl,fr=f(left),f(right)
        if fl==0 or fr==0 or mp.sign(fl)==mp.sign(fr):return None
        for _ in range(1400):
            mid=(left+right)/2
            if mid==left or mid==right:break
            fm=f(mid)
            if fm==0:return +mid
            if mp.sign(fm)==mp.sign(fl):left,fl=mid,fm
            else:right=mid
            if right-left<min(abs(mid),abs(1-mid))*mp.mpf('1e-70'):break
        return +(left+right)/2


def pair_crossings(a,b,c,d,lo=.02,hi=.5):
    if (a,b)==(c,d):return {'identical':True,'roots':[]}
    if not ((a>c and b>d) or (a<c and b<d)):
        return {'identical':False,'roots':[]}
    fl=_pair_diff(a,b,c,d,lo);fr=_pair_diff(a,b,c,d,hi)
    close=max(abs(a-c),abs(b-d))<1e-8
    precise=None
    if close or min(abs(fl),abs(fr))<1e-11 or hi-lo<1e-8 or min(lo,1-hi)<1e-8:
        precise=_mp_crossing(a,b,c,d,lo,hi)
        root=None if precise is None else float(precise)
    elif fl*fr<0:
        u=float(brentq(lambda u:_pair_diff(a,b,c,d,lo+u*(hi-lo)),0.,1.,xtol=5e-15,rtol=1e-14))
        root=float(lo+u*(hi-lo))
    else:root=None
    if root is None:return {'identical':False,'roots':[]}
    if not lo<root<hi and precise is None:
        precise=_mp_crossing(a,b,c,d,lo,hi)
        if precise is None:return {'identical':False,'roots':[]}
        root=float(precise)
    # A genuine interior root can round to a float endpoint. Keep its existence
    # and high-precision location explicit instead of dropping it or crashing.
    unresolved=not lo<root<hi
    result={'identical':False,'roots':[root]}
    if unresolved and precise is not None:
        result.update(float_resolution_unresolved=True,root_decimal=mp.nstr(precise,80))
    return result


def _segment_winners(a,b,labels,left,right):
    p=(left+right)/2
    if p in (left,right) or right-left<1e-10:
        with mp.workdps(80):
            mid=(mp.mpf(float(left))+mp.mpf(float(right)))/2
            values=[_mp_metric(ai,bi,mid) for ai,bi in zip(a,b)];best=max(values)
            return tuple(label for label,v in zip(labels,values) if best-v<mp.mpf('1e-65'))
    scores=macro_f1(a,b,p)
    near=np.flatnonzero(scores.max()-scores<1e-10)
    if len(near)==1:return (labels[int(near[0])],)
    # Midpoint ties are assessed at high precision; tolerance is only a trigger
    # for refinement, not a rule merging distinct near-optimal policies.
    with mp.workdps(80):
        values=[_mp_metric(a[i],b[i],p) for i in near];best=max(values)
        return tuple(labels[int(i)] for i,v in zip(near,values) if best-v<mp.mpf('1e-65'))


def phase_diagram(a,b,labels=None,lo=.02,hi=.5):
    a,b=np.asarray(a),np.asarray(b); labels=list(range(len(a))) if labels is None else labels
    cuts=[lo,hi];unresolved=[]
    for i in range(len(a)):
        for j in range(i):
            crossing=pair_crossings(a[i],b[i],a[j],b[j],lo,hi)
            cuts.extend(crossing['roots'])
            if crossing.get('float_resolution_unresolved'):
                unresolved.append({'policies':[labels[j],labels[i]],'root_decimal':crossing['root_decimal']})
    cuts=np.array(sorted(set(cuts)))
    segments=[]
    for left,right in zip(cuts[:-1],cuts[1:]):
        winners=_segment_winners(a,b,labels,left,right)
        if segments and segments[-1]['winners']==winners:
            segments[-1]['right']=float(right)
        else:segments.append({'left':float(left),'right':float(right),'winners':winners})
    boundaries=[]
    for p in [lo]+[s['right'] for s in segments]:
        scores=macro_f1(a,b,p)
        boundaries.append({'p':float(p),'winners':[labels[i] for i in np.flatnonzero(np.abs(scores-scores.max())<1e-9)]})
    return {'segments':segments,'boundaries':boundaries,'pairwise_cut_count':len(cuts),
            'float_resolution_unresolved':unresolved}


def frontier_phase_diagram(a,b,labels=None,lo=.02,hi=.5):
    """Sorted ROC-frontier stack, using the proved single-crossing property.

    Groups only exactly identical floating-point ROC pairs. Root solves use
    Brent's method with 80-digit bisection for near-coincident/end-point cases;
    output is numerical, not interval-arithmetic certified. Boundary tie lists
    use a separate 1e-9 display tolerance, not a statistical certificate.
    """
    a,b=np.asarray(a),np.asarray(b)
    if len(a)==0 or not 0<lo<hi<1:raise ValueError('Nonempty pool and interior interval required')
    labels=list(range(len(a))) if labels is None else list(labels)
    grouped={}
    for i in range(len(a)):grouped.setdefault((float(a[i]),float(b[i])),[]).append(labels[i])
    points=sorted(grouped,key=lambda x:(x[1],-x[0]))
    frontier=[];maxa=-np.inf
    for ai,bi in points:
        if ai>maxa:
            frontier.append((ai,bi,tuple(grouped[(ai,bi)])));maxa=ai
    stack=[];starts=[];solves=0;unresolved=[]
    def diff(new,old,p):return _pair_diff(new[0],new[1],old[0],old[1],p)
    for candidate in frontier:
        skip=False
        while stack:
            old=stack[-1]
            if diff(candidate,old,hi)<=0:skip=True;break
            if diff(candidate,old,lo)>=0:entry=lo
            else:
                crossing=pair_crossings(candidate[0],candidate[1],old[0],old[1],lo,hi)
                if not crossing['roots']:raise ArithmeticError('Bracketed crossing could not be resolved')
                entry=crossing['roots'][0];solves+=1
                if crossing.get('float_resolution_unresolved'):
                    unresolved.append({'policies':[old[2],candidate[2]],'root_decimal':crossing['root_decimal']})
            if entry<=starts[-1]:stack.pop();starts.pop()
            else:break
        if not skip:
            if not stack:entry=lo
            stack.append(candidate);starts.append(entry)
    segments=[{'left':float(start),'right':float(starts[j+1] if j+1<len(starts) else hi),
               'winners':item[2]} for j,(start,item) in enumerate(zip(starts,stack))]
    segments=[s for s in segments if s['left']<s['right']]
    boundaries=[]
    # Include every original label attaining the numerical maximum at each
    # transition/end, including a frontier policy that has zero-length tenure.
    for p in [lo]+[s['right'] for s in segments]:
        values=macro_f1(a,b,p)
        boundaries.append({'p':float(p),'winners':[labels[i] for i in np.flatnonzero(abs(values-values.max())<1e-9)]})
    return {'segments':segments,'boundaries':boundaries,'frontier_size':len(frontier),'root_solves':solves,
            'float_resolution_unresolved':unresolved}


def rates(y,scores,thresholds):
    y=np.asarray(y); scores=np.asarray(scores); thresholds=np.asarray(thresholds)
    if len(np.unique(y))!=2:raise ValueError('Both classes required')
    pred=scores>=thresholds
    return pred[y==1].mean(axis=0),pred[y==0].mean(axis=0)


def bounds(a,b,n1,n0,alpha=.05,kind='dkw',n_models=1,constants=None):
    a,b=np.asarray(a),np.asarray(b); k=len(a)
    if min(n1,n0)<=0:raise ValueError('Both classes required')
    if kind=='dkw':
        e1=np.sqrt(np.log(4*n_models/alpha)/(2*n1));e0=np.sqrt(np.log(4*n_models/alpha)/(2*n0))
        al,au=np.clip(a-e1,0,1),np.clip(a+e1,0,1)
        bl,bu=np.clip(b-e0,0,1),np.clip(b+e0,0,1)
    elif kind=='cp':
        tail=alpha/(4*k)
        def cp(x,n):
            count=np.rint(x*n).astype(int)
            return (np.where(count==0,0,beta.ppf(tail,count,n-count+1)),
                    np.where(count==n,1,beta.ppf(1-tail,count+1,n-count)))
        al,au=cp(a,n1);bl,bu=cp(b,n0)
    else:raise ValueError(kind)
    if constants is not None:
        # -1 all-negative, +1 all-positive, 0 general. Never infer constants
        # just because all observed predictions in one sample are identical.
        constants=np.asarray(constants)
        for mask,value in [(constants==-1,0.),(constants==1,1.)]:
            al[mask]=au[mask]=bl[mask]=bu[mask]=value
    return al,au,bl,bu


def metric_bounds(bound,p):
    al,au,bl,bu=bound
    return macro_f1(al,bu,p),macro_f1(au,bl,p)


def recommendations(a,b,bound,ps,groups=None):
    p=np.asarray(ps)[None,:]
    point=macro_f1(np.asarray(a)[:,None],np.asarray(b)[:,None],p)
    low,up=metric_bounds(tuple(v[:,None] for v in bound),p)
    selected=low.argmax(axis=0); empirical=point.argmax(axis=0)
    col=np.arange(p.shape[1]);others=up.copy();others[selected,col]=-np.inf
    cert=low[selected,col]>others.max(axis=0)+1e-12
    regret_bound=np.clip(up.max(axis=0)-low[selected,col],0,1)
    result=dict(selected=selected,empirical=empirical,certified=cert,
                lower=low,upper=up,regret_bound=regret_bound)
    if groups is not None:
        groups=np.asarray(groups);other_group=up.copy()
        other_group[groups[:,None]==groups[selected][None,:]]=-np.inf
        result['learner_certified']=low[selected,col]>other_group.max(axis=0)+1e-12
    return result


def derivative_bound(a,b,lo,hi):
    a,b=np.asarray(a),np.asarray(b)
    qlo=b+(1+a-b)*lo; qhi=b+(1+a-b)*hi
    # dF/dp = ab/q² - (1-a)(1-b)/(2-q)².
    return a*b/np.minimum(qlo,qhi)**2+(1-a)*(1-b)/np.minimum(2-qlo,2-qhi)**2


def interval_regret_bound(bound,selected,lo,hi,points=401):
    """A conservative continuum bound: grid max + explicit Lipschitz correction.

    Bound is valid mathematically on the simultaneous rate event; ordinary
    floating-point arithmetic is used, not machine-verified interval arithmetic.
    """
    ps=np.linspace(lo,hi,points); al,au,bl,bu=bound
    low,up=metric_bounds(tuple(v[:,None] for v in bound),ps[None,:])
    gap=up.max(axis=0)-low[selected]
    lip=float(derivative_bound(au,bl,lo,hi).max()+derivative_bound(al[selected],bu[selected],lo,hi))
    correction=lip*(hi-lo)/(2*(points-1))
    return {'bound':float(min(1,gap.max()+correction)), 'grid_max':float(gap.max()),
            'lipschitz_correction':float(correction),'grid_points':points}


def choose_threshold(y,p,metric):
    order=np.argsort(-p,kind='stable');q=p[order];z=y[order]
    ends=np.r_[np.flatnonzero(q[:-1]!=q[1:]),len(q)-1]
    tp=np.r_[0,np.cumsum(z)[ends]].astype(float);fp=np.r_[0,ends+1-np.cumsum(z)[ends]].astype(float)
    fn=float(z.sum())-tp;tn=float(len(z)-z.sum())-fp
    ts=np.r_[np.nextafter(float(q[0]),np.inf),q[ends]]
    if metric=='macro_f1':v=tp/np.maximum(2*tp+fp+fn,1)+tn/np.maximum(2*tn+fp+fn,1)
    else:v=.5*(tp/(tp+fn)+tn/(tn+fp))
    best=np.flatnonzero(np.abs(v-v.max())<=1e-12)
    return float(ts[best[np.lexsort((ts[best],np.abs(ts[best]-.5)))[0]]])
