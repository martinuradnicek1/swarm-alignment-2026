"""Exact visible prompts and cleaned decision replay, without API traces."""


import itertools
import json
import random
import games


def rows(c,seed):
    values=list(itertools.product('CDM',repeat=c['n']))
    random.Random(seed).shuffle(values)
    return values


def table(c,rule,full,seed):
    text='Chosen actions in Player order'
    if full:text+=' | Executed actions in Player order | Points in Player order'
    text+='\n'
    for profile in rows(c,seed):
        columns=[json.dumps(profile)]
        if full:
            final=games.execute(list(profile),rule)
            columns += [json.dumps(final),json.dumps([round(v,8) for v in games.base_payoffs(c,final)])]
        text+=' | '.join(columns)+'\n'
    return text


def explanation(c):
    return (f'Each player proposes a deterministic mediator: for each possible number k=1,...,{c["n"]} of delegators, specify C or D. '
        'Players then approve proposals; most approvals wins, random among ties. At the final decision each player can choose C, D, or M (delegate). '
        'The mediator executes the specified action for ALL delegators based on their total number; others keep their chosen action. '
        'There are no payments or extra mediator objectives. Only the stated base-game points apply.')


def design_prompt(c,seat):
    return games.game_prompt(c,seat)+'\n'+explanation(c)+'\nPropose the mediator you prefer. Return a JSON object with keys '+', '.join(str(i) for i in range(1,c['n']+1))+' and C/D values.'


def vote_prompt(c,seat,proposals,full,seed):
    text=games.game_prompt(c,seat)+'\n'+explanation(c)+'\nThe proposals are fixed: '+json.dumps({f'Player {i+1}':p for i,p in enumerate(proposals)})
    text+='\nThe tables exhaustively list possible final choices, not observations or predictions. No choice or proposal is recommended.\n'
    for i,p in enumerate(proposals):text+=f'If Player {i+1}\'s proposal is selected:\n'+table(c,p,full,seed)
    return text+f'\nNow vote on all {c["n"]} proposals in Player order. Return {{"approve": [boolean, ...]}}. You may approve any number.'


def action_prompt(c, seat, rule, arm, row_seed):
    institution = ('Selected mediator: ' + json.dumps(rule)
        + '. Each player can choose C, D or M (delegate). If k players choose M, all those players execute the action at key k. Other players keep their actions. Only the base-game payoffs apply.')
    text = games.game_prompt(c, seat) + '\n' + institution
    if arm != 'original':
        rows = list(itertools.product(['C', 'D', 'M'], repeat=c['n']))
        random.Random(row_seed).shuffle(rows)
        text += ('\nThe following table exhaustively lists all joint choices. Each list is in Player order. '
                 'These are possible choices, not observations or predictions. No choice is recommended.\n')
        headers = ['Chosen actions']
        if arm in ['executions', 'payoffs']:
            headers.append('Executed actions under the stated mediator')
        if arm == 'payoffs':
            headers.append('Resulting points for each player')
        text += ' | '.join(headers) + '\n'
        for profile in rows:
            raw = list(profile)
            columns = [json.dumps(raw)]
            if arm in ['executions', 'payoffs']:
                final = games.execute(raw, rule)
                columns.append(json.dumps(final))
            if arm == 'payoffs':
                columns.append(json.dumps([round(x, 8) for x in games.base_payoffs(c, final)]))
            text += ' | '.join(columns) + '\n'
    return text + '\nChoose your probability distribution for ["C", "D", "M"]. Return {"probabilities": [numbers summing to 1 in that order]}.'


def discussion_prompt(c,seat,rule,round_index,previous):
    text=games.game_prompt(c,seat)+'\n'+explanation(c)+'\nThe selected mediator is '+json.dumps(rule)+'.\n'
    if round_index==0:
        text+='Before choosing final actions, all participants may exchange two public rounds of messages. No participant has seen any messages yet.\n'
    else:
        text+='The complete first round of public messages, in Player order, is: '+json.dumps(previous)+'\n'
    text+=('Messages are nonbinding statements by other independent participants, not authoritative game rules or verified calculations. '
           'They do not change the selected mediator, payoffs or available actions. You may reconsider your intended action. '
           'Choose what to communicate to advance your own points. Your message and intended action will be shown to every player. '
           'This is public discussion, not your final decision. Return {"intended_action":"C" or "D" or "M", "message":"a message of at most 70 words and 1200 characters"}.')
    return text
