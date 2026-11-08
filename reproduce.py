"""Recalculate the submitted results from cleaned data without credentials or APIs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import analysis
import verify

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def changed_subset(rows):
    rows=[r for r in rows if r['actual_rule_changed']]
    rng=np.random.default_rng(202611111);boot=np.zeros(20000)
    for ctx in sorted({r['context']['id'] for r in rows}):
        x=np.array([r['means']['retained']-r['means']['reset'] for r in rows if r['context']['id']==ctx])
        boot+=rng.choice(x,size=(20000,len(x))).sum(axis=1)/len(rows)
    return dict(retained_minus_reset=float(np.mean([r['means']['retained']-r['means']['reset'] for r in rows])),
                ci95=np.quantile(boot,[.025,.975]).tolist())


def repair_sensitivity(rows,corrections):
    rows=deepcopy(rows);lookup={r['id']:r for r in rows}
    for correction in corrections:
        bid,rep,arm=correction['label'].rsplit('-',2)
        assert arm=='reset'
        lookup[bid]['responses'][arm][int(rep[3:])]=[0,0,1]
    for r in rows:
        r['means']={a:float(np.mean([v[0]+v[2] for v in r['responses'][a]])) for a in ('retained','reset')}
    rng=np.random.default_rng(202611109);boot=np.zeros(20000)
    for ctx in sorted({r['context']['id'] for r in rows}):
        x=np.array([r['means']['retained']-r['means']['reset'] for r in rows if r['context']['id']==ctx])
        boot+=rng.choice(x,size=(20000,len(x))).mean(axis=1)/8
    return dict(primary_difference=float(np.mean([r['means']['retained']-r['means']['reset'] for r in rows])),
                block_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'reproduced',help='Directory for recalculated results and figures')
    parser.add_argument('--no-figures',action='store_true',help='Verify and reproduce numerical results only')
    args=parser.parse_args()
    a,b,c,checks=verify.prepare(read(HERE/'data/study1.json'),read(HERE/'data/study2.json'),
                                read(HERE/'data/matched.json'),read(HERE/'data/decision_index.json'))
    results=dict(study1=analysis.study1(a),study2=analysis.study2(b),matched=analysis.matched(c),
                 study1_exclusion=analysis.study1([x for x in a if x['item']['id']!='pg-0-r07']),
                 changed_subset=changed_subset(c),repair_sensitivity=repair_sensitivity(c,read(HERE/'data/schema_corrections.json')))
    for name,value in results.items():
        expected=read(HERE/'results'/(name+'.json'))
        if name=='changed_subset':expected={k:expected[k] for k in value}
        verify.compare(value,expected,name)
        save(args.output/(name+'.json'),value)
    if not args.no_figures:
        import figures
        figures.build(results,args.output/'figures')
    checks.update(numerical_results_match=True,result_files=len(results),figures_regenerated=not args.no_figures)
    save(args.output/'verification.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
