"""Independent identities/edge cases for the scientific prototype."""
import json,unittest
from pathlib import Path
import numpy as np
import sympy as sp
from sklearn.metrics import f1_score
from policy_certificate import *

class ScientificProperties(unittest.TestCase):
    def test_confusion_matrix_identity(self):
        rng=np.random.default_rng(7)
        for _ in range(100):
            tp,fn,fp,tn=rng.integers(1,300,size=4);y=np.r_[np.ones(tp+fn),np.zeros(fp+tn)]
            pred=np.r_[np.ones(tp),np.zeros(fn),np.ones(fp),np.zeros(tn)]
            self.assertAlmostEqual(macro_f1(tp/(tp+fn),fp/(fp+tn),(tp+fn)/len(y)),f1_score(y,pred,average='macro'),places=13)
    def test_symbolic_cubic(self):
        p,a,b,c,d=sp.symbols('p a b c d');q1=b+(1+a-b)*p;q2=d+(1+c-d)*p
        e1=b+(1-a-b)*p;e2=d+(1-c-d)*p
        numerator=sp.Poly(sp.expand(e2*q1*(2-q1)-e1*q2*(2-q2)),p)
        self.assertEqual(numerator.degree(),3)
    def test_known_crossing_and_ties(self):
        # A zero-error constant-negative beats constant-positive for p<.5.
        cross=pair_crossings(0,0,1,1,.01,.99);self.assertAlmostEqual(cross['roots'][0],.5,places=12)
        self.assertTrue(pair_crossings(.8,.2,.8,.2)['identical'])
        phase=phase_diagram([0,1],[0,1],['negative','positive'],.01,.99)
        self.assertEqual(len(phase['segments']),2)
        self.assertEqual(set(phase['boundaries'][1]['winners']),{'negative','positive'})
    def test_phase_matches_dense_oracle(self):
        rng=np.random.default_rng(8)
        for _ in range(25):
            a=rng.uniform(size=6);b=rng.uniform(size=6);phase=phase_diagram(a,b,lo=.01,hi=.99)
            for seg in phase['segments']:
                ps=np.linspace(seg['left'],seg['right'],11)[1:-1]
                for p in ps:self.assertIn(int(macro_f1(a,b,p).argmax()),seg['winners'])
    def test_atoms_tie_and_known_constants(self):
        y=np.array([0,1,0,1,0,1]);score=np.array([.1,.1,.5,.5,.9,.9])
        threshold=choose_threshold(y,score,'macro_f1')
        candidates=np.r_[np.unique(score),np.nextafter(score.max(),np.inf)]
        value=f1_score(y,score>=threshold,average='macro')
        self.assertAlmostEqual(value,max(f1_score(y,score>=t,average='macro') for t in candidates))
        for kind in ['cp','dkw']:
            bd=bounds(np.array([0,1,1]),np.array([0,1,0]),3,3,kind=kind,constants=[-1,1,0])
            self.assertEqual(bd[0][0],bd[1][0]);self.assertEqual(bd[2][1],bd[3][1])
            self.assertLess(bd[0][2],1) # observed perfect prediction is not known population perfection
    def test_envelope_monotonicity_and_continuum_bound(self):
        rng=np.random.default_rng(9)
        for _ in range(20):
            a=rng.uniform(size=5);b=rng.uniform(size=5);bd=bounds(a,b,100,400)
            ps=np.linspace(.02,.7,10001);l,u=metric_bounds(tuple(v[:,None] for v in bd),ps[None,:]);truth=macro_f1(a[:,None],b[:,None],ps[None,:])
            self.assertTrue(np.all(l<=truth+1e-12));self.assertTrue(np.all(truth<=u+1e-12))
            bound=interval_regret_bound(bd,0,.02,.7,points=41)
            self.assertGreaterEqual(bound['bound']+1e-12,min(1,(u.max(axis=0)-l[0]).max()))
    def test_prevalence_endpoints_rejected(self):
        with self.assertRaises(ValueError):macro_f1(.8,.2,0)
    def test_learner_vs_policy_certificate(self):
        a=np.array([.95,.95,.5]);b=np.array([.05,.05,.5]);bd=(a,a,b,b)
        rec=recommendations(a,b,bd,[.2],groups=[0,0,1])
        self.assertFalse(rec['certified'][0]);self.assertTrue(rec['learner_certified'][0])
    def test_frontier_stack_and_single_crossing(self):
        rng=np.random.default_rng(20)
        for _ in range(50):
            a=rng.uniform(size=16);b=rng.uniform(size=16)
            a=np.r_[a,0,1,a[0]];b=np.r_[b,0,1,b[0]]
            fast=frontier_phase_diagram(a,b,lo=.01,hi=.99)
            slow=phase_diagram(a,b,lo=.01,hi=.99)
            self.assertLessEqual(len(fast['segments']),fast['frontier_size'])
            for p in np.linspace(.0101,.9899,101):
                i=macro_f1(a,b,p).argmax()
                seg=next(z for z in fast['segments'] if z['left']<=p<=z['right'])
                self.assertIn(int(i),seg['winners'])
            self.assertEqual(len(fast['segments']),len(slow['segments']))
            for f,s in zip(fast['segments'],slow['segments']):self.assertAlmostEqual(f['right'],s['right'],places=8)
        for a,b,c,d in [(1,1,0,0),(.7,.9,0,0),(1,1,.3,.2),(1,.7,.6,0)]:
            crossings=pair_crossings(a,b,c,d,1e-9,1-1e-9)['roots'];self.assertEqual(len(crossings),1)
            x=crossings[0];self.assertLess(macro_f1(a,b,x/2),macro_f1(c,d,x/2))
            self.assertGreater(macro_f1(a,b,(1+x)/2),macro_f1(c,d,(1+x)/2))

if __name__=='__main__':unittest.main(verbosity=2)
