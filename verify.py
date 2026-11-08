"""Offline validation of cleaned decisions, prompt fingerprints and exact payoffs."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import itertools
import json
import random
import games
import prompts


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def pay(c,actions):
    if c['game']=='pd':
        return [F(str(c['R'])) if a==b=='C' else F(str(c['S'])) if a=='C' else
                F(str(c['T'])) if b=='C' else F(str(c['P'])) for a,b in [actions,actions[::-1]]]
    shared=F(str(c['multiplier']))*actions.count('C')/c['n']
    return [shared+int(a=='D') for a in actions]


def exact_score(c,rule,distributions):
    value=F(0)
    for indices in itertools.product(range(len(distributions[0])),repeat=c['n']):
        probability=F(1)
        for i,j in enumerate(indices):
            probability*=F(str(distributions[i][j]))
        actions=['CDM'[j] for j in indices]; k=actions.count('M')
        final=[rule[str(k)] if a=='M' else a for a in actions] if rule else actions
        value+=probability*sum(pay(c,final))/c['n']
    return float((value-F(str(c['lo'])))/(F(str(c['hi']))-F(str(c['lo']))))


def prepare(study1,study2,matched,index):
    seen=set(); checked=0; scores1=[]; scores2=[]
    def check(records):
        nonlocal checked
        for row in records:
            key=row['key'];assert key not in seen;seen.add(key)
            expected=index[key]
            assert row['model']==expected['model'] and row['kind']==expected['kind']
            assert digest(row['messages'])==expected['messages_sha256'], 'Prompt mismatch: '+key
            assert digest(row['answer'])==expected['answer_sha256'], 'Answer mismatch: '+key
            checked+=1
    for b in study1:
        c=b['item']['context'];assert b['models']==games.roster(c,b['item']['rep'])
        values={}
        for v in ('0','1'):
            vote=b['votes'][v];answers=vote['answers'];rule=vote['selected_rule']
            if None in answers:
                assert b['item']['id']=='pg-0-r07' and v=='0' and answers[1] is None and vote['winner'] is None
                for unknown in itertools.product([False,True],repeat=c['n']):
                    complete=[list(unknown) if x is None else x for x in answers]
                    assert b['proposals'][games.choose(complete,b['item']['seed']+2)]==rule
            else:
                assert games.choose(answers,b['item']['seed']+2)==vote['winner']
                assert b['proposals'][vote['winner']]==rule
            assert games.quality(c,rule)==vote['qualifying']
        for arm,o in b['outcomes'].items():
            rule=b['votes'][arm[1]]['selected_rule']
            value=exact_score(c,rule,o['distributions'])
            assert abs(value-o['score'])<1e-9 and abs(value-games.score(c,rule,o['distributions'])['score'])<1e-9
            values[arm]=value
        ct=b['contract'];values['contract']=exact_score(c,None,ct['distributions'])
        assert abs(values['contract']-ct['expected_normalized_group_payoff'])<1e-9
        assert abs(values['contract']-games.expected_contract_score(c,ct['distributions']))<1e-9
        rule=ct['proposals'][ct['winner']]
        changes=games.transfer(c,ct['executed_actions'],rule) if all(ct['signatures']) else [0]*c['n']
        assert all(abs(a-b)<1e-8 for a,b in zip(changes,ct['transfers']))
        assert all(abs(float(x)+t-y)<1e-8 for x,t,y in zip(pay(c,ct['executed_actions']),changes,ct['final_payoffs']))
        assert [games.quality(c,p) for p in b['proposals']]==b['proposal_quality']
        scores1.append(dict(b,values=values));check(prompts.study1_records(b))
    for b in study2:
        c=b['item']['context'];assert b['models']==games.roster(c,b['item']['rotation'])
        assert games.choose(b['votes'],b['item']['seed']+2)==b['winner']
        assert b['proposals'][b['winner']]==b['selected_rule']
        assert games.quality(c,b['selected_rule'])==b['qualifying']
        assert [games.quality(c,p) for p in b['proposals']]==b['proposal_quality']
        values={}
        for arm,o in b['outcomes'].items():
            value=exact_score(c,b['selected_rule'],o['distributions'])
            assert abs(value-o['score'])<1e-9;values[arm]=value
        scores2.append(dict(b,values=values));check(prompts.study2_records(b))
    lookup={b['item']['id']:b for b in study2}; matched=deepcopy(matched)
    for row in matched:
        b=lookup[row['id']];c=b['item']['context'];i=b['models'].index('gemma')
        assert row['context']==c
        assert row['actual_rule_changed']==(b['selected_rule']!=games.rule_for(c,'unconditional_cooperation'))
        for peers in itertools.product('CDM',repeat=c['n']-1):
            values=[]
            for own in 'CDM':
                raw=list(peers);raw.insert(i,own)
                values.append(pay(c,['C' if a=='M' else a for a in raw])[i])
            assert values[1]>max(values[0],values[2])
        row['means']={a:sum(v[0]+v[2] for v in row['responses'][a])/8 for a in ('retained','reset')}
        check(prompts.matched_records(row,b))
    assert len(study1)==128 and len(study2)==len(matched)==64
    assert seen==set(index) and checked==6303
    return scores1,scores2,matched,dict(passed=True,cleaned_decisions_verified=checked,
        exact_prompt_and_answer_fingerprints=True,independent_payoff_recalculation=True,
        missing_vote_invariance=True,network_calls=0)


def compare(actual,expected,path='result'):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),path
        for key in expected:compare(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+f'[{i}]')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert abs(actual-expected)<1e-10,(path,actual,expected)
    else:
        assert actual==expected,(path,actual,expected)
