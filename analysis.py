"""Paired block analyses extracted from the frozen analyses; no network access."""


import math
import numpy as np


STUDY1_ARMS=['v0a0','v0a1','v1a0','v1a1']
STUDY2_ARMS=['r0','r1','h0','h1']
MATCHED_ARMS=['retained','reset']


def study1(blocks):
    rng=np.random.default_rng(202611092);keys=STUDY1_ARMS+['contract'];results={}
    contrasts=[('v1a1','v0a0'),('v1a1','v0a1'),('v0a1','v0a0'),('v1a0','v0a0'),('v1a1','v1a0'),('v1a1','contract')]
    for family in ['all','pd','pg']:
        rr=[b for b in blocks if family=='all' or b['item']['context']['game']==family]
        contexts=sorted({b['item']['context']['id'] for b in rr})
        data=np.array([[b['values'][k] for k in keys] for b in rr])
        boot=np.zeros((10000,len(keys)));means=np.zeros(len(keys));weights=[]
        for c in contexts:
            ids=[i for i,b in enumerate(rr) if b['item']['context']['id']==c]
            pick=rng.choice(ids,size=(10000,len(ids)),replace=True)
            boot+=data[pick].mean(axis=1)/len(contexts)
            means+=data[ids].mean(axis=0)/len(contexts)
            weights.extend([1/(len(contexts)*len(ids))]*len(ids))
        cells={k:dict(mean=float(means[i]),ci95=np.quantile(boot[:,i],[.025,.975]).tolist()) for i,k in enumerate(keys)}
        cs=[]
        for a,b in contrasts:
            ia,ib=keys.index(a),keys.index(b);d=data[:,ia]-data[:,ib];bs=boot[:,ia]-boot[:,ib]
            primary=(family=='all' and (a,b)==('v1a1','v0a0')) or (family=='pd' and (a,b)==('v1a1','v0a1'))
            alpha=.025 if primary else .05;mean=float(means[ia]-means[ib]);radius=math.sqrt(2*math.log(2/alpha)*sum(w*w for w in weights))
            cs.append(dict(a=a,b=b,primary=primary,mean=mean,ci=np.quantile(bs,[alpha/2,1-alpha/2]).tolist(),
                           confidence=1-alpha,bounded_ci=[max(-1,mean-radius),min(1,mean+radius)]))
        interaction=boot[:,3]-boot[:,2]-boot[:,1]+boot[:,0]
        results[family]=dict(blocks=len(rr),cells=cells,contrasts=cs,interaction=dict(mean=float(means[3]-means[2]-means[1]+means[0]),ci95=np.quantile(interaction,[.025,.975]).tolist()),
            proposal_available=sum(any(b['proposal_quality']) for b in rr),selected_qualifying={v:sum(b['votes'][v]['qualifying'] for b in rr) for v in ['0','1']},
            controls={model:dict(decisions=sum(b['models'].count(model) for b in rr),strict_D=sum(d==[0,1,0] for b in rr for m,d in zip(b['models'],b['controls']) if m==model),mean_D=float(np.mean([d[1] for b in rr for m,d in zip(b['models'],b['controls']) if m==model]))) for model in ['gemma','sol']},
            unanimous_delegation={a:float(np.mean([b['outcomes'][a]['unanimous_delegation'] for b in rr])) for a in STUDY1_ARMS})
    return results


def study2(blocks):
    rng=np.random.default_rng(202611102); results={}; contrasts=[('h1','h0'),('r1','r0'),('h0','r0'),('h1','r1')]
    for family in ['all','pd','pg']:
        rows=[b for b in blocks if family=='all' or b['item']['context']['game']==family]
        contexts=sorted({b['item']['context']['id'] for b in rows}); data=np.array([[b['values'][a] for a in STUDY2_ARMS] for b in rows])
        boot=np.zeros((10000,4)); means=np.zeros(4); weights=[]
        for ctx in contexts:
            ids=[i for i,b in enumerate(rows) if b['item']['context']['id']==ctx]
            ix=rng.choice(ids,size=(10000,len(ids)),replace=True); boot+=data[ix].mean(axis=1)/len(contexts)
            means+=data[ids].mean(axis=0)/len(contexts); weights += [1/(len(contexts)*len(ids))]*len(ids)
        cells={a:dict(mean=float(means[i]),ci95=np.quantile(boot[:,i],[.025,.975]).tolist()) for i,a in enumerate(STUDY2_ARMS)}
        cs=[]
        for a,b in contrasts:
            ia,ib=STUDY2_ARMS.index(a),STUDY2_ARMS.index(b); mean=float(means[ia]-means[ib]); samples=boot[:,ia]-boot[:,ib]
            radius=math.sqrt(2*math.log(40)*sum(w*w for w in weights))
            cs.append(dict(a=a,b=b,primary=family=='all' and (a,b)==('h1','h0'),mean=mean,ci95=np.quantile(samples,[.025,.975]).tolist(),
                bounded_ci95=[max(-1,mean-radius),min(1,mean+radius)]))
        inter=boot[:,3]-boot[:,2]-boot[:,1]+boot[:,0]
        results[family]=dict(blocks=len(rows),cells=cells,contrasts=cs,
            interaction=dict(mean=float(means[3]-means[2]-means[1]+means[0]),ci95=np.quantile(inter,[.025,.975]).tolist()),
            proposal_available=sum(any(b['proposal_quality']) for b in rows),selected_qualifying=sum(b['qualifying'] for b in rows),
            controls={model:dict(decisions=sum(b['models'].count(model) for b in rows),strict_D=sum(d==[0,1,0] for b in rows for m,d in zip(b['models'],b['controls']) if m==model),
                mean_D=float(np.mean([d[1] for b in rows for m,d in zip(b['models'],b['controls']) if m==model]))) for model in ['gemma','sol']},
            unanimous_delegation={a:float(np.mean([b['outcomes'][a]['unanimous_delegation'] for b in rows])) for a in STUDY2_ARMS})
    return results


def matched(rows):
    result = {}
    for family in ['all', 'pd', 'pg']:
        selected = [r for r in rows if family == 'all' or r['context']['game'] == family]
        contexts = sorted(set(r['context']['id'] for r in selected))
        values = np.array([[r['means'][a] for a in MATCHED_ARMS] for r in selected])
        rng = np.random.default_rng(202611109)
        boot = np.zeros((20000, 2))
        for ctx in contexts:
            indices = [i for i, r in enumerate(selected) if r['context']['id'] == ctx]
            sample = rng.choice(indices, size=(20000, len(indices)), replace=True)
            boot += values[sample].mean(axis=1) / len(contexts)
        mean = values.mean(axis=0)
        diff = float(mean[0] - mean[1])
        radius = math.sqrt(2 * math.log(40) / len(selected))
        cell = {}
        for j, arm in enumerate(MATCHED_ARMS):
            vectors = [v for r in selected for v in r['responses'][arm]]
            cell[arm] = dict(non_D_probability=float(mean[j]), ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),
                             responses=len(vectors), pure_C=sum(v==[1,0,0] for v in vectors),
                             pure_D=sum(v==[0,1,0] for v in vectors), pure_M=sum(v==[0,0,1] for v in vectors),
                             mixtures=sum(max(v)<1 for v in vectors),
                             blocks_with_variable_answers=sum(len({tuple(v) for v in r['responses'][arm]})>1 for r in selected))
        primary = dict(mean=diff, ci95=np.quantile(boot[:,0]-boot[:,1],[.025,.975]).tolist(),
                       bounded_ci95=[max(-1,diff-radius),min(1,diff+radius)], primary=family=='all')
        if family == 'all':
            rng = np.random.default_rng(202611110)
            deltas = values[:,0]-values[:,1]
            null = (rng.choice([-1,1],size=(100000,len(selected)))*deltas).mean(axis=1)
            primary['block_sign_swap_p_two_sided'] = float((1+np.sum(np.abs(null)>=abs(diff)-1e-12))/100001)
        result[family] = dict(blocks=len(selected), cells=cell, retained_minus_reset=primary)
    result['descriptive_rule_subsets'] = {
        str(changed): dict(blocks=sum(r['actual_rule_changed']==changed for r in rows),
                          **{a:float(np.mean([r['means'][a] for r in rows if r['actual_rule_changed']==changed])) for a in MATCHED_ARMS})
        for changed in [False,True]}
    return result
