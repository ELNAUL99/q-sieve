#!/usr/bin/env python3
"""Q-SIEVE: reproducible synthetic demonstration, not a banking model.

Run: python model.py --output model-data.json
Tests: python model.py --test

Eight features are consolidated from actual generated transaction rows. One
sample is one independent account/window. No real customer data is used.
The NumPy statevector applies the same gates as Qiskit's ZZFeatureMap with
reps=2, entanglement='linear': H on every qubit, P(2*x_i), and CX-P(2*(pi-x_i)
*(pi-x_j))-CX on adjacent pairs in each repetition. Global phase is irrelevant
to fidelity. Source: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.ZZFeatureMap

Hyperparameters and transformations are fixed before looking at the held-out
test split. Sigmoid scaling is fitted only on a separate calibration split.
The deployment-like 90%-recall threshold is selected on calibration only;
held-out recall is measured and is not guaranteed to reach 90%.
"""

from __future__ import annotations
import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix

VERSION = 'qsieve-synthetic-zz8-v1.0'
SEED = 314159
WINDOW_HOURS = 72
TOY_THRESHOLD = 10000
CLEAR_MAX = 0.20
ESCALATE_MIN = 0.75
FEATURES = [
    {'key':'velocity','label':'Activity vs baseline','unit':'×','description':'Transaction count divided by the account’s synthetic expected count for this 72-hour window.'},
    {'key':'average_amount','label':'Average transfer','unit':'EUR','description':'Gross absolute transaction volume divided by transaction count.'},
    {'key':'inbound_count_share','label':'Inbound count share','unit':'share','description':'Credit count divided by all transactions; preserves flow direction.'},
    {'key':'inbound_value_share','label':'Inbound value share','unit':'share','description':'Total credits divided by gross absolute volume.'},
    {'key':'net_to_gross','label':'Net / gross flow','unit':'ratio','description':'Signed (credits − debits) / (credits + debits). Near zero indicates pass-through; it does not establish wrongdoing.'},
    {'key':'dwell_hours','label':'Median next-debit delay','unit':'hours','description':'Median hours from each credit to the next later debit in the window. Timing proxy only; funds are not matched.'},
    {'key':'counterparties','label':'Distinct counterparties','unit':'count','description':'Unique synthetic sending and receiving counterparties in this account window.'},
    {'key':'threshold_share','label':'Near demo threshold','unit':'share','description':'Share of transactions at €9,000–€9,999.99. €10,000 is a toy scenario parameter, not a reporting-law claim.'},
]
KEYS = [f['key'] for f in FEATURES]
REGIMES = ['Inbound fragmented','Outbound fragmented','Inbound near-threshold','Outbound near-threshold']
PROFILES = ['Online marketplace seller','Contractor payroll service','Property services company','Community event organiser','Wholesale distributor','Independent consultancy']


def consolidate(transactions, baseline):
    """Compute all eight classifier features from rows and one account baseline."""
    incoming = [t for t in transactions if t['direction'] == 'in']
    outgoing = [t for t in transactions if t['direction'] == 'out']
    credits = sum(t['amount'] for t in incoming)
    debits = sum(t['amount'] for t in outgoing)
    gross = credits + debits
    out_times = sorted(t['hour'] for t in outgoing)
    delays = [next((h-t['hour'] for h in out_times if h >= t['hour']), WINDOW_HOURS-t['hour']) for t in incoming]
    features = {
        'velocity':len(transactions)/baseline,
        'average_amount':gross/len(transactions),
        'inbound_count_share':len(incoming)/len(transactions),
        'inbound_value_share':credits/gross,
        'net_to_gross':(credits-debits)/gross,
        'dwell_hours':float(np.median(delays)),
        'counterparties':len({t['counterparty'] for t in transactions}),
        'threshold_share':sum(0.9*TOY_THRESHOLD <= t['amount'] < TOY_THRESHOLD for t in transactions)/len(transactions),
    }
    return features, credits, debits


def make_window(rng, i, split, label=None, regime=None):
    """Generate one labelled scenario; latent class drives overlapping behaviours.

    Labels are invented generator truth, never an allegation about a person.
    Busy legitimate cases occur in every amount/direction regime. A 20% overlap
    component deliberately makes features insufficient for perfect classification.
    """
    label = int(rng.random() < 0.30) if label is None else int(label)
    regime = int(rng.integers(4)) if regime is None else int(regime)
    near = regime >= 2
    inbound = regime % 2 == 0
    many = int(rng.integers(10,25) if near else rng.integers(70,180))
    few = int(rng.integers(2,7))
    n_in, n_out = (many,few) if inbound else (few,many)
    overlap = rng.random() < .20
    if overlap:
        net_magnitude = rng.uniform(.04,.40)
        dwell = rng.uniform(1.0,12.0)
        repeat_fraction = rng.uniform(.3,.85)
    elif label:
        net_magnitude = rng.uniform(.003,.12)
        dwell = rng.uniform(.03,1.8)
        repeat_fraction = rng.uniform(.55,.95)
    else:
        net_magnitude = rng.uniform(.18,.67)
        dwell = rng.uniform(6,32)
        repeat_fraction = rng.uniform(.10,.65)
    # Positive net means retained funds, negative means inventory/payroll drawdown.
    net_ratio = net_magnitude * (1 if inbound else -1)
    if near:
        if label and not overlap:
            dominant_amounts = rng.uniform(9300,9990,many)
        else:
            dominant_amounts = rng.uniform(7800,10600,many)
    else:
        dominant_amounts = np.clip(rng.lognormal(np.log(420),.52,many),70,1900)
    dominant_sum = float(dominant_amounts.sum())
    if inbound:
        total_in = dominant_sum
        total_out = total_in*(1-net_ratio)/(1+net_ratio)
        in_amounts = dominant_amounts
        out_amounts = rng.dirichlet(np.full(n_out,8.0))*total_out
    else:
        total_out = dominant_sum
        total_in = total_out*(1+net_ratio)/(1-net_ratio)
        in_amounts = rng.dirichlet(np.full(n_in,8.0))*total_in
        out_amounts = dominant_amounts
    # Same window contains collection then payment; difference is actual delay.
    in_times = np.sort(rng.uniform(0,1,n_in))
    first_out = float(in_times[-1]+dwell)
    out_times = np.r_[first_out, np.sort(rng.uniform(first_out,first_out+2,n_out-1))]
    cp_pool = max(2,int((n_in+n_out)*repeat_fraction))
    account_id = f'{split.upper()}-{i:04d}'
    start = datetime(2026,9,28,tzinfo=timezone.utc)
    rows = []
    for direction,amounts,times in [('in',in_amounts,in_times),('out',out_amounts,out_times)]:
        for j,(amount,hour) in enumerate(zip(amounts,times)):
            rows.append({'id':f'{account_id}-{direction}-{j:03d}','hour':float(hour),
                         'timestamp':(start+timedelta(hours=float(hour))).isoformat().replace('+00:00','Z'),
                         'direction':direction,'amount':round(float(amount),2),
                         'counterparty':f'CP-{account_id}-{int(rng.integers(cp_pool)):03d}'})
    rows.sort(key=lambda t:t['hour'])
    velocity = float(rng.uniform(8,42))
    baseline = len(rows)/velocity
    features,credits,debits = consolidate(rows,baseline)
    return {'id':account_id,'regime':REGIMES[regime],'regimeIndex':regime,
            'profile':PROFILES[int(rng.integers(len(PROFILES)))],
            'baselineExpectedCount':baseline,'count':len(rows),'inflow':round(credits,2),
            'outflow':round(debits,2),'gross':round(credits+debits,2),'features':features,
            'transactions':rows,'syntheticTruth':label,'split':split}


def create_split(seed,n,split):
    rng = np.random.default_rng(seed)
    # Fixed prevalence and all four regimes, independently shuffled per split.
    labels = np.array([1]*int(.30*n)+[0]*(n-int(.30*n)))
    rng.shuffle(labels)
    return [make_window(rng,i,split,label=int(labels[i]),regime=i%4) for i in range(n)]


def feature_matrix(windows):
    return np.array([[w['features'][k] for k in KEYS] for w in windows],dtype=float)


def normalized_features(x):
    """Fixed domain transforms (no fitting on evaluation data) into [0,1]."""
    z = np.array(x,dtype=float,copy=True)
    z[:,0] = np.log1p(z[:,0])/np.log1p(50)
    z[:,1] = np.log1p(z[:,1])/np.log1p(20000)
    z[:,4] = (z[:,4]+1)/2
    z[:,5] = np.log1p(z[:,5])/np.log1p(WINDOW_HOURS)
    z[:,6] = np.log1p(z[:,6])/np.log1p(200)
    return np.clip(z,0,1)


def hadamard_all(state,qubits):
    for q in range(qubits):
        stride = 1 << q
        for base in range(0,1 << qubits,2*stride):
            a = state[:,base:base+stride].copy()
            b = state[:,base+stride:base+2*stride].copy()
            state[:,base:base+stride] = (a+b)/np.sqrt(2)
            state[:,base+stride:base+2*stride] = (a-b)/np.sqrt(2)
    return state


def zz_states(angles,reps=2):
    """Exact batched statevector of adjacent-pair ZZFeatureMap gates."""
    angles = np.atleast_2d(angles)
    m,n = angles.shape
    indices = np.arange(1 << n)
    bits = ((indices[:,None] >> np.arange(n)) & 1).astype(float)
    phase = 2*angles @ bits.T
    for q in range(n-1):
        parity = np.logical_xor(bits[:,q],bits[:,q+1]).astype(float)
        phase += 2*((np.pi-angles[:,q])*(np.pi-angles[:,q+1]))[:,None]*parity
    diagonal = np.exp(1j*phase)
    states = np.zeros((m,1 << n),dtype=complex)
    states[:,0] = 1
    for _ in range(reps):
        hadamard_all(states,n)
        states *= diagonal
    return states


def fidelity(a,b):
    return np.clip(np.abs(a @ b.conj().T)**2,0,1)


def route(score,clear_max=CLEAR_MAX,escalate_min=ESCALATE_MIN):
    if score < clear_max:
        return 'clear'
    if score >= escalate_min:
        return 'escalate'
    return 'review'


def can_auto_clear(score,existing_status='open'):
    """A suspicious/escalated status cannot be closed by automatic action."""
    return existing_status not in {'suspicious','escalated','escalate'} and route(score) == 'clear'


def threshold_at_recall(y,score,target=.9):
    # Largest threshold retaining at least target on calibration positives.
    positive = np.sort(np.asarray(score)[np.asarray(y)==1])[::-1]
    return float(positive[int(math.ceil(target*len(positive)))-1])


def metric_record(name,model_id,y,score,cal_y,cal_score,train_seconds,test_seconds):
    threshold = threshold_at_recall(cal_y,cal_score)
    keep = score >= threshold
    clear = score < CLEAR_MAX
    n_pos = int(np.sum(y==1))
    n_neg = int(np.sum(y==0))
    cleared_pos = int(np.sum(clear & (y==1)))
    return {'id':model_id,'name':name,'prAuc':float(average_precision_score(y,score)),
            'prAucDefinition':'Average precision (step-weighted PR area), positive = synthetic suspicious.',
            'recallAt90Policy':float(np.mean(keep[y==1])),
            'falsePositiveEliminationAt90Policy':float(np.mean(~keep[y==0])),
            'policy90Threshold':threshold,'calibrationRecallAt90Policy':float(np.mean(cal_score[cal_y==1]>=threshold)),
            'autoClearCoverage':float(np.mean(clear)),
            'autoClearError':float(cleared_pos/clear.sum()) if clear.any() else None,
            'autoClearedSuspicious':cleared_pos,'autoClearedCount':int(clear.sum()),
            'retainedRecall':float(1-cleared_pos/n_pos),'positiveTest':n_pos,'negativeTest':n_neg,
            'trainSeconds':train_seconds,'testSeconds':test_seconds,
            'confusionAt90Policy':{'truePositive':int(np.sum(keep & (y==1))),
               'falsePositive':int(np.sum(keep & (y==0))),'trueNegative':int(np.sum(~keep & (y==0))),
               'falseNegative':int(np.sum(~keep & (y==1)))}}


def evidence_for(w):
    f=w['features']
    flow = 'Funds largely pass through' if abs(f['net_to_gross'])<.15 else 'Material funds retained or drawn down'
    return [
        {'feature':'net_to_gross','title':flow,'value':f"{f['net_to_gross']:+.1%}",'detail':'Signed net change divided by gross flow. A near-zero value is a timing/context signal, not proof of laundering.'},
        {'feature':'dwell_hours','title':'Credit-to-next-debit timing','value':f"{f['dwell_hours']:.1f} h",'detail':'Median time to the next debit; this does not trace the same funds or establish causation.'},
        {'feature':'inbound_count_share','title':'Direction survives aggregation','value':f"{f['inbound_count_share']:.0%} inbound",'detail':f"{w['regime']}; credits represent {f['inbound_value_share']:.0%} of value."},
        {'feature':'threshold_share','title':'Amount pattern','value':f"{f['threshold_share']:.0%} near threshold",'detail':'Toy €10,000 threshold used for synthetic scenarios; not a statutory rule.'},
    ]


def literal_gate_reference(angle,reps=2):
    """Independent literal H/P/CX implementation for simulator verification."""
    n=len(angle); s=np.zeros(1<<n,dtype=complex);s[0]=1
    for _ in range(reps):
        for q in range(n):
            for j in range(1<<n):
                if not (j>>q)&1:
                    k=j^(1<<q);a,b=s[j],s[k];s[j]=(a+b)/np.sqrt(2);s[k]=(a-b)/np.sqrt(2)
        for q in range(n):
            for j in range(1<<n):
                if (j>>q)&1:s[j]*=np.exp(2j*angle[q])
        for q in range(n-1):
            def cx():
                for j in range(1<<n):
                    if (j>>q)&1 and not (j>>(q+1))&1:
                        k=j^(1<<(q+1));s[j],s[k]=s[k],s[j]
            cx()
            for j in range(1<<n):
                if (j>>(q+1))&1:s[j]*=np.exp(2j*(np.pi-angle[q])*(np.pi-angle[q+1]))
            cx()
    return s


def tests():
    rng=np.random.default_rng(89)
    angles=rng.uniform(0,np.pi,(16,8));states=zz_states(angles);kernel=fidelity(states,states)
    assert np.allclose(np.sum(np.abs(states)**2,axis=1),1,atol=1e-12)
    assert np.allclose(kernel,kernel.T,atol=1e-12)
    assert np.allclose(np.diag(kernel),1,atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min()>-1e-10
    assert np.allclose(states[0],literal_gate_reference(angles[0]),atol=1e-12)
    assert route(.199999)=='clear' and route(.2)=='review' and route(.75)=='escalate'
    assert can_auto_clear(.1) and not can_auto_clear(.9)
    assert not can_auto_clear(.01,'suspicious') and not can_auto_clear(.01,'escalated')
    splits=[create_split(SEED+j,20,name) for j,name in enumerate(['train','calibration','test'])]
    ids=[{w['id'] for w in s} for s in splits]
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    w=splits[0][0];f,credit,debit=consolidate(w['transactions'],w['baselineExpectedCount'])
    assert np.isclose(f['net_to_gross'],(credit-debit)/(credit+debit))
    assert np.isclose(f['inbound_value_share'],credit/(credit+debit))
    assert threshold_at_recall(np.ones(10),np.arange(10)/10)==.1
    print('PASS: literal-gate equivalence; state normalization; kernel symmetry/PSD/diagonal; policy boundaries; suspicious-close guard; disjoint splits; feature identities; calibration recall threshold.')


def build(output):
    tests()
    train=create_split(SEED,240,'train')
    calibration=create_split(SEED+1,120,'calibration')
    test=create_split(SEED+2,160,'test')
    demo_pool=create_split(SEED+3,120,'demo')
    all_windows=train+calibration+test+demo_pool
    x=normalized_features(feature_matrix(all_windows))
    y=np.array([w['syntheticTruth'] for w in all_windows])
    nt,nc,nv=len(train),len(calibration),len(test)
    c=slice(nt,nt+nc);t=slice(nt+nc,nt+nc+nv);d=slice(nt+nc+nv,None)
    a=time.perf_counter();states=zz_states(x*np.pi);state_time=time.perf_counter()-a
    a=time.perf_counter();ktrain=fidelity(states[:nt],states[:nt]);kcal=fidelity(states[c],states[:nt]);kernel_train_time=time.perf_counter()-a
    a=time.perf_counter();ktest=fidelity(states[t],states[:nt]);test_kernel_time=time.perf_counter()-a
    kdemo=fidelity(states[d],states[:nt])
    model_specs=[('quantum','ZZ fidelity + SVC',SVC(kernel='precomputed',C=4,class_weight='balanced'),ktrain,kcal,ktest,kdemo),
                 ('rbf','RBF SVC',SVC(kernel='rbf',C=4,gamma='scale',class_weight='balanced'),x[:nt],x[c],x[t],x[d]),
                 ('histgb','Histogram gradient boosting',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=9,l2_regularization=1,random_state=SEED),x[:nt],x[c],x[t],x[d])]
    try:
        from xgboost import XGBClassifier
        model_specs.append(('xgboost','XGBoost',XGBClassifier(n_estimators=100,max_depth=3,learning_rate=.08,subsample=1,colsample_bytree=1,random_state=SEED,n_jobs=1),x[:nt],x[c],x[t],x[d]))
    except ImportError:
        pass
    metrics=[];quantum_scores=None;quantum_model=None;quantum_scaler=None
    for mid,name,model,tx,cx,vx,dx in model_specs:
        a=time.perf_counter();model.fit(tx,y[:nt]);fit_seconds=time.perf_counter()-a
        decision=lambda z:model.decision_function(z) if hasattr(model,'decision_function') else np.log(np.clip(model.predict_proba(z)[:,1],1e-7,1-1e-7)/np.clip(1-model.predict_proba(z)[:,1],1e-7,1))
        # A separate sigmoid scaling step; scores are illustrative, not validated risk probabilities.
        cal_decision=decision(cx)
        scaler=LogisticRegression(C=1,solver='lbfgs').fit(cal_decision[:,None],y[c])
        cal_score=scaler.predict_proba(cal_decision[:,None])[:,1]
        a=time.perf_counter();test_score=scaler.predict_proba(decision(vx)[:,None])[:,1];score_seconds=time.perf_counter()-a
        if mid=='quantum':
            fit_seconds+=kernel_train_time+state_time*(nt+nc)/len(all_windows)
            score_seconds+=test_kernel_time+state_time*nv/len(all_windows)
            quantum_scores=scaler.predict_proba(decision(dx)[:,None])[:,1]
            quantum_model=model;quantum_scaler=scaler
        metrics.append(metric_record(name,mid,y[t],test_score,y[c],cal_score,fit_seconds,score_seconds))
    # Select a varied presentation queue from a separate demo-only pool.
    # This never influences held-out evaluation. Include extremes and uncertain cases.
    chosen=[]
    for regime in range(4):
        candidates=[i for i,w in enumerate(demo_pool) if w['regimeIndex']==regime]
        for target in [0,.5,1]:
            candidates.sort(key=lambda i:abs(quantum_scores[i]-target))
            chosen.append(candidates[0])
            candidates.remove(candidates[0])
    names=['Harbour Collective','Lumi Market','Cedar Transfer Desk','Northline Studios','Atelier Supply','Baltic Dispatch','Aurora Rentals','Kallio Wholesale','Silver Birch Trade','Sisu Events','Tammisto Services','Meridian Payments']
    cases=[]
    for j,i in enumerate(chosen):
        w=demo_pool[i]
        w['name']=names[j]+' (synthetic)'
        w['riskScore']=float(quantum_scores[i]);w['status']=route(w['riskScore'])
        w['evidence']=evidence_for(w)
        near=np.argsort(kdemo[i])[::-1][:3]
        w['precedents']=[{'id':train[k]['id'],'label':'synthetic suspicious' if train[k]['syntheticTruth'] else 'synthetic legitimate',
                          'similarity':float(kdemo[i,k]),'regime':train[k]['regime'],
                          'summary':f"Generator-labelled {'suspicious' if train[k]['syntheticTruth'] else 'legitimate'} reference; {train[k]['count']} transfers, {train[k]['features']['dwell_hours']:.1f} h next-debit delay."} for k in near]
        w['syntheticTruthNote']='Generator ground truth for this fictional demonstration; not an analyst disposition.'
        cases.append(w)
    cases.sort(key=lambda w:w['riskScore'],reverse=True)
    result={
      'meta':{'title':'Q-SIEVE','modelVersion':VERSION,'generatedAt':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'seed':SEED,'dataMode':'Self-generated synthetic account windows',
        'qubits':8,'reps':2,'entanglement':'linear','simulation':'Exact NumPy statevector, 256 complex amplitudes; no quantum hardware run',
        'kernel':'K(x,z) = |⟨φ(x)|φ(z)⟩|²','thresholdAmount':TOY_THRESHOLD,'currency':'EUR','windowHours':WINDOW_HOURS,
        'scoreDescription':'Sigmoid-scaled SVC decision score using a separate synthetic calibration split. Not a validated probability of financial crime.',
        'fixtureSelection':'12 varied illustrative windows selected by score from a separate 120-window synthetic demo pool. Demo selection does not affect test metrics.',
        'caveats':['Synthetic patterns and 30% suspicious prevalence are invented; results do not estimate real bank performance.',
                   'The kernel is computed on a classical simulator. No speedup, sample-efficiency gain or quantum advantage is established.',
                   'Near-zero net flow and short next-debit timing can occur in legitimate businesses.',
                   '€10,000 is a toy scenario parameter and not a statement about reporting law.',
                   'Evidence cards describe observable features; they are not causal model attribution.'],
        'sources':[{'title':'Supplied Q-SIEVE Team Onboarding, Oct 1 2026','url':'attached PDF'},
                   {'title':'IBM Qiskit ZZFeatureMap gate convention','url':'https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.ZZFeatureMap'}]},
      'features':FEATURES,
      'policy':{'clearMax':CLEAR_MAX,'escalateMin':ESCALATE_MIN,'recallTarget':.9,
        'thresholdBasis':'Default demonstration route cutoffs fixed before evaluation: clear < 0.20, escalate ≥ 0.75, otherwise human review. They are not certified safe.',
        'closureRule':'Only eligible clean-score cases may be auto-cleared. An escalated/suspicious disposition blocks automatic closure regardless of score.',
        'autoClearEnabledByDefault':False},
      'evaluation':{'counts':{'train':nt,'calibration':nc,'test':nv,'demoPool':len(demo_pool),
                   'positiveTrain':int(y[:nt].sum()),'positiveCalibration':int(y[c].sum()),'positiveTest':int(y[t].sum())},
          'models':metrics,'protocol':'Independent account windows; fixed train/calibration/test seeds. Same features, same splits. Hyperparameters fixed; no test-set tuning. Timings are this local run only.',
          'metricDefinitions':{'prAuc':'Average precision on held-out synthetic suspicious cases.',
             'recallAt90Policy':'Test suspicious cases retained / all test suspicious cases, using threshold chosen to retain ≥90% of calibration positives.',
             'falsePositiveEliminationAt90Policy':'Test legitimate cases below that threshold / all test legitimate cases. Measures low-score alert elimination only, separate from the three-way demo route.',
             'autoClearCoverage':'Test cases below 0.20 / all test cases.',
             'autoClearError':'Synthetic suspicious cases below 0.20 / all test cases below 0.20.',
             'retainedRecall':'1 − (synthetic suspicious cases below 0.20 / all test suspicious cases).'},
          'labelAblation':{'status':'Not run','plannedPositives':[50,100,200,500],'reason':'This prototype has 72 labelled training positives; the full source-proposed benchmark requires more validated data and repeated splits.'},
          'hardware':{'status':'Not run','reason':'No hardware credentials or jobs used; no simulator-to-hardware claims.'},
          'xgboost':{'status':'included' if any(m['id']=='xgboost' for m in metrics) else 'not installed','substitute':'Histogram gradient boosting is labelled explicitly and is not XGBoost.'},
          'caveats':['Single-seed synthetic sanity check, not an AML benchmark or production safety claim.',
                     '30% synthetic suspicious prevalence is deliberately enriched and unlike a live bank queue.',
                     'A 90% target on calibration does not guarantee 90% recall on unseen data.',
                     'Auto-clear defaults are demonstrative; audit and routing do not eliminate classification errors.',
                     'Baselines use fixed reasonable parameters, not exhaustive tuning; full-strength baseline selection remains future work.']},
      'cases':cases,
      'model':{'supportVectorCount':int(len(quantum_model.support_)),'trainingIds':[w['id'] for w in train],
               'calibrationIds':[w['id'] for w in calibration],'testIds':[w['id'] for w in test],
               'fixedFeatureTransforms':'log1p for velocity/amount/delay/counterparties; fixed domain bounds; net/gross shifted from [-1,1]; clip all to [0,1], multiply by π.',
               'svc':{'C':4,'classWeight':'balanced','kernel':'precomputed'},
               'sigmoid':{'coefficient':float(quantum_scaler.coef_[0,0]),'intercept':float(quantum_scaler.intercept_[0])},
               'reproducibility':'python model.py --output model-data.json regenerates windows, kernels, scores and metrics; machine-dependent timings differ.'}}
    Path(output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'counts':result['evaluation']['counts'],'models':metrics,'routes':{s:sum(w['status']==s for w in cases) for s in ['clear','review','escalate']}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(Path(__file__).with_name('model-data.json')));parser.add_argument('--test',action='store_true');args=parser.parse_args()
    tests() if args.test else build(args.output)
