"""Reconstruct the original visible inputs from cleaned experimental records."""
import json
import games
from prompts_base import action_prompt, design_prompt, discussion_prompt, vote_prompt


def initial():
    return [dict(role='system',content=games.SYSTEM)]


def user(text):
    return dict(role='user',content=text)


def assistant(value):
    return dict(role='assistant',content=json.dumps(value))


def record(study,label,model,kind,messages,answer):
    return dict(key=study+'/'+label,model=model,kind=kind,messages=messages,answer=answer)


def contract_records(b):
    c=b['item']['context']; n=c['n']; ct=b['contract']; label=b['item']['id']+'-contract'
    explanation=(f'Each player first proposes an enforceable payment contract: numbers for C and D in [-{c["transfer_bound"]},{c["transfer_bound"]}]. For an action X, its actor receives the proposed amount for X, paid equally by the other players. Negative amounts reverse that transfer. Transfers apply for every player and sum to zero. Players then approve proposals; the one with most approvals wins, random among ties. Every player must then sign it or NO transfers apply. Finally everyone chooses C or D. Propose the contract you prefer. Return {{"C": number, "D": number}}.')
    def one(i,stage,text,kind,answer):
        return record('study1',label+f'-p{i}-{stage}',ct['models'][i],kind,
                      initial()+[user(games.game_prompt(c,i)+'\n'+text)],answer)
    for i in range(n):
        yield one(i,'design',explanation,'contract',ct['proposals'][i])
    public=json.dumps({f'Player {i+1}':x for i,x in enumerate(ct['proposals'])})
    vote=explanation+'\nThe proposals are fixed: '+public+f'\nNow vote on all {n} proposals in Player order. Return {{"approve": [boolean, ...]}}. You may approve any number.'
    for i in range(n):
        yield one(i,'vote',vote,'votes',dict(approve=ct['votes'][i]))
    selected=ct['proposals'][ct['winner']]
    signing=explanation+'\nThe selected contract is '+json.dumps(selected)+'. Now decide whether to sign it. All must sign for transfers to apply. Return {"sign": true} or {"sign": false}.'
    for i in range(n):
        yield one(i,'sign',signing,'sign',dict(sign=ct['signatures'][i]))
    institution='Selected contract: '+json.dumps(selected)+'. Signatures in Player order: '+json.dumps(ct['signatures'])+'. '+('All signed. Transfers WILL apply to every action as described: the actor receives the action-specific amount, funded equally by all other players.' if all(ct['signatures']) else 'At least one rejected. NO contract transfers apply; use only base payoffs.')
    act=institution+'\nChoose your probability distribution for '+json.dumps(['C','D'])+'. Return {"probabilities": [numbers summing to 1 in that order]}.'
    for i in range(n):
        yield one(i,'act',act,'action',dict(probabilities=ct['distributions'][i]))


def study1_records(b):
    c=b['item']['context']; seed=b['item']['seed']; bid=b['item']['id']
    def one(i,stage,prompt,kind,answer):
        return record('study1',bid+f'-{stage}-p{i}',b['models'][i],kind,initial()+[user(prompt)],answer)
    for i in range(c['n']):
        yield one(i,'design',design_prompt(c,i),'mediator',b['proposals'][i])
    for v in (0,1):
        votes=b['votes'][str(v)]
        for i in range(c['n']):
            if votes['answers'][i] is not None:
                yield one(i,f'v{v}-vote',vote_prompt(c,i,b['proposals'],bool(v),seed),'votes',dict(approve=votes['answers'][i]))
        for a in (0,1):
            arm=f'v{v}a{a}'
            for i in range(c['n']):
                yield one(i,arm+'-act',action_prompt(c,i,votes['selected_rule'],'payoffs' if a else 'profiles',seed),
                          'action_mediator',dict(probabilities=b['outcomes'][arm]['distributions'][i]))
    yield from contract_records(b)
    for i in range(c['n']):
        yield one(i,'bad-control',action_prompt(c,i,games.rule_for(c,'unconditional_cooperation'),'payoffs',seed),
                  'action_mediator',dict(probabilities=b['controls'][i]))


def history(b,i):
    c=b['item']['context']; seed=b['item']['seed']
    result=initial()+[user(design_prompt(c,i)),assistant(b['proposals'][i]),
                      user(vote_prompt(c,i,b['proposals'],False,seed)),assistant(dict(approve=b['votes'][i]))]
    for rnd in range(2):
        result += [user(discussion_prompt(c,i,b['selected_rule'],rnd,b['discussion'][0] if rnd else None)),
                   assistant(b['discussion'][rnd][i])]
    return result


def final_control(b,i):
    c=b['item']['context']
    announcement=('The complete second public round, in Player order, is: '+json.dumps(b['discussion'][1])+'.\n'
        'CHANGED FINAL RULE: before actions are chosen, the selected mediator is replaced by the rule stated below. '
        'This rule supersedes the earlier selected mediator and all discussion about it. Previous messages and intentions are nonbinding. '
        'There are no future rewards or penalties for changing your intention. Choose your actual final action under the NEW authoritative rule.\n')
    return user(announcement+action_prompt(c,i,games.rule_for(c,'unconditional_cooperation'),'payoffs',b['item']['seed']))


def study2_records(b):
    c=b['item']['context']; seed=b['item']['seed']; bid=b['item']['id']
    for i in range(c['n']):
        h=history(b,i)
        stages=[('design',h[:2],'mediator',b['proposals'][i]),
                ('vote',h[:4],'votes',dict(approve=b['votes'][i])),
                ('message0',h[:6],'speech',b['discussion'][0][i]),
                ('message1',h[:8],'speech',b['discussion'][1][i])]
        for stage,messages,kind,answer in stages:
            yield record('study2',bid+f'-{stage}-p{i}',b['models'][i],kind,messages,answer)
        for workflow in ('r','h'):
            for a in (0,1):
                arm=workflow+str(a)
                announcement=('The complete second public round, in Player order, is: '+json.dumps(b['discussion'][1])+'.\n'
                    'The public messages remain nonbinding and do not change the authoritative rules. Now make your actual final decision.\n') if workflow=='h' else ''
                prompt=announcement+action_prompt(c,i,b['selected_rule'],'payoffs' if a else 'profiles',seed)
                messages=(h if workflow=='h' else initial())+[user(prompt)]
                yield record('study2',bid+f'-{arm}-act-p{i}',b['models'][i],'action_mediator',messages,
                             dict(probabilities=b['outcomes'][arm]['distributions'][i]))
        yield record('study2',bid+f'-changed-control-p{i}',b['models'][i],'action_mediator',
                     h+[final_control(b,i)],dict(probabilities=b['controls'][i]))


def matched_records(row,source):
    i=source['models'].index('gemma')
    final=final_control(source,i)
    messages={'retained':history(source,i)+[final], 'reset':initial()+[final]}
    for rep in range(8):
        for arm in ('retained','reset'):
            yield record('matched',f'{row["id"]}-rep{rep:02d}-{arm}','gemma','action_mediator',messages[arm],
                         dict(probabilities=row['responses'][arm][rep]))
