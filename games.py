"""Game definitions and exact expected-payoff calculations used by the release."""


import hashlib
import itertools
import json
import math
import random


SYSTEM = 'You are an independent participant in a finite decision game. Your objective is to maximize your own points under the stated rules. Other participants are separate decision-makers. Return only the requested JSON object. You may reason before answering. Do not invent observations or additional rules.'


def digest(obj):
 return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def contexts():
 out = []
 for index, temptation in enumerate([3.5, 4.0, 4.5, 5.5]):
  out.append({'id': f'pd-{index}', 'game': 'pd', 'n': 2, 'T': temptation, 'R': 3, 'P': 1, 'S': 0, 'lo': 1, 'hi': 3, 'transfer_bound': 12})
 for index, multiplier in enumerate([1.2, 1.5, 2.0, 2.7]):
  out.append({'id': f'pg-{index}', 'game': 'pg', 'n': 3, 'multiplier': multiplier, 'lo': 1, 'hi': multiplier, 'transfer_bound': 6})
 return out


def base_payoffs(c, actions):
 assert len(actions) == c['n'] and all(a in ['C', 'D'] for a in actions)
 if c['game'] == 'pg':
  pool = c['multiplier']*actions.count('C')/c['n']
  return [pool+(a == 'D') for a in actions]
 a, b = actions
 return [c['R'], c['R']] if a == b == 'C' else [c['P'], c['P']] if a == b == 'D' else [c['S'], c['T']] if a == 'C' else [c['T'], c['S']]


def transfer(c, actions, contract):
 n = c['n']; amounts = [contract[a] for a in actions]
 changes = [a-(sum(amounts)-a)/(n-1) for a in amounts]
 assert abs(sum(changes)) < 1e-8
 return changes


def game_prompt(c, seat):
 if c['game'] == 'pd':
  description = f"Two players independently choose C or D. Payoffs (Player 1, Player 2): CC=({c['R']},{c['R']}), CD=({c['S']},{c['T']}), DC=({c['T']},{c['S']}), DD=({c['P']},{c['P']})."
 else:
  description = f"Three players each have 1 point and independently choose C (contribute it) or D (keep it). If k players choose C, EVERY player receives {c['multiplier']}*k/3 points from the pool. A D player additionally keeps their 1 point; a C player does not."
 return f"You are Player {seat+1} of {c['n']}. {description}\nThis is a new independent encounter; there are no later rewards, other interactions or hidden enforcement. Actions are chosen simultaneously at the final step."


def valid(obj, kind, c):
 n = c['n']
 if kind == 'contract':
  if set(obj) != {'C','D'} or any(type(v) not in (float,int) or not math.isfinite(v) or abs(v)>c['transfer_bound'] for v in obj.values()):
   raise ValueError('Return exactly C and D with finite numeric transfers within the stated bounds')
 elif kind == 'mediator':
  if set(obj) != {str(i) for i in range(1,n+1)} or any(v not in ['C','D'] for v in obj.values()):
   raise ValueError('Return every delegator-count key with a C or D action')
 elif kind == 'votes':
  if set(obj) != {'approve'} or type(obj['approve']) != list or len(obj['approve']) != n or any(type(x) != bool for x in obj['approve']):
   raise ValueError('Return approve: a boolean list of the stated length')
 elif kind == 'sign':
  if set(obj) != {'sign'} or type(obj['sign']) != bool: raise ValueError('Return sign: true or false')
 elif kind.startswith('action'):
  size = 3 if kind == 'action_mediator' else 2
  v = obj.get('probabilities')
  if set(obj) != {'probabilities'} or type(v) != list or len(v) != size or any(type(x) not in (float,int) or not math.isfinite(x) or x<0 or x>1 for x in v) or not math.isclose(sum(v),1,abs_tol=1e-6):
   raise ValueError(f'Return probabilities: {size} finite numbers in [0,1] summing to 1')
 return obj


def rule_for(c, name):
    return {str(k): 'C' if name == 'unconditional_cooperation' or k == c['n'] else 'D'
            for k in range(1, c['n'] + 1)}


def execute(raw, rule):
    k = raw.count('M')
    return [rule[str(k)] if a == 'M' else a for a in raw]


def choose(votes,seed):
    n=len(votes);counts=[sum(v[i] for v in votes) for i in range(n)]
    priority=list(range(n));random.Random(seed).shuffle(priority)
    return next(i for i in priority if counts[i]==max(counts))


def quality(c,rule):
    if rule[str(c['n'])]!='C':return False
    for seat in range(c['n']):
        for others in itertools.product('CDM',repeat=c['n']-1):
            values=[]
            for own in 'CDM':
                profile=list(others);profile.insert(seat,own)
                values.append(base_payoffs(c,execute(profile,rule))[seat])
            if values[2]+1e-9<max(values[:2]):return False
    return True


def score(c,rule,distributions):
    value=0.;unanimous=1.
    for d in distributions:unanimous*=d[2]
    for inds in itertools.product(range(3),repeat=c['n']):
        probability=1.
        for seat,a in enumerate(inds):probability*=distributions[seat][a]
        raw=['CDM'[i] for i in inds]
        value+=probability*sum(base_payoffs(c,execute(raw,rule)))/c['n']
    return dict(score=(value-c['lo'])/(c['hi']-c['lo']),unanimous_delegation=unanimous)


def roster(c, rotation):
    models = ['sol'] * c['n']
    models[rotation % c['n']] = 'gemma'
    return models


def expected_contract_score(c, distributions):
    value = 0.0
    for indices in itertools.product(range(2), repeat=c['n']):
        probability = math.prod(distributions[i][a] for i, a in enumerate(indices))
        value += probability * sum(base_payoffs(c, ['CD'[i] for i in indices])) / c['n']
    return (value-c['lo'])/(c['hi']-c['lo'])
