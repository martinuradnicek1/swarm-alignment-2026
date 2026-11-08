"""Scientific figures generated exclusively from released results."""
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'swarm-alignment-mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def export(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):
        fig.savefig(out/(name+'.'+extension),dpi=240,facecolor='white')
    plt.close(fig)


def build(results,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.titlesize':11,'axes.labelsize':10,'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained',sharey=True)
    d=results['study2']
    for ax,family,title in zip(axes,['pd','pg'],['Two-player dilemma','Three-player shared pool']):
        for j,(suffix,label) in enumerate([('0','Choices listed'),('1','Consequences shown')]):
            cells=[d[family]['cells'][prefix+suffix] for prefix in ['r','h']]
            x=np.arange(2)+(j-.5)*.18
            y=np.array([c['mean'] for c in cells]);lo=np.array([c['ci95'][0] for c in cells]);hi=np.array([c['ci95'][1] for c in cells])
            ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='o',
                        color=['#66717e','#137c85'][j],capsize=4,markersize=6,lw=1.5,label=label)
        ax.set_xticks([0,1],['Fresh final\ncontext','Earlier history\n+ discussion'])
        ax.set_xlim(-.35,1.35);ax.set_ylim(-.04,1.05);ax.set_title(title+'\n32 matched cases',pad=8)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Group score (0 to 1)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Collective performance after shared-rule selection',fontsize=12)
    export(fig,out,'collective-performance')

    fig,ax=plt.subplots(figsize=(7.2,2.55))
    fig.subplots_adjust(left=.29,right=.98,top=.84,bottom=.26)
    labels=['Overall (primary)\n64 cases','Two-player (descriptive)\n32 cases','Public goods (descriptive)\n32 cases']
    ax.axvline(0,color='#7b8790',lw=1,linestyle='--')
    for y,family in zip([2,1,0],['all','pd','pg']):
        c=results['matched'][family]['retained_minus_reset'];mean=100*c['mean'];low,high=[100*x for x in c['ci95']]
        color='#137c85' if family=='all' else '#66717e'
        ax.errorbar(mean,y,xerr=[[mean-low],[high-mean]],fmt='o',color=color,capsize=4,markersize=7 if family=='all' else 6,lw=1.7)
        ax.text(mean,y+.19,f'{mean:.1f} [{low:.1f}, {high:.1f}]',ha='center',va='bottom',fontsize=9,
                color=color,weight='bold' if family=='all' else 'normal')
    ax.set_yticks([2,1,0],labels);ax.set_ylim(-.42,2.7);ax.set_xlim(-20,40)
    ax.set_xticks([-20,-10,0,10,20,30,40]);ax.tick_params(axis='y',length=0,pad=10)
    ax.set_xlabel('Increase in disadvantageous responses (percentage points)',labelpad=8)
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
    fig.suptitle('Effect of retaining earlier interaction context',x=.54,y=.99,fontsize=12)
    export(fig,out,'retained-context')

    fig,ax=plt.subplots(figsize=(7.2,1.8));fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.03)
    ax.set_xlim(0,10);ax.set_ylim(0,2.3);ax.axis('off')
    xs=[.05,2.58,5.11,7.64];w=2.25
    for i,(x,label) in enumerate(zip(xs,['Propose a rule','Vote on a rule','Discuss plans','Choose an action'])):
        ax.add_patch(FancyBboxPatch((x,1.25),w,.65,boxstyle='round,pad=0.015,rounding_size=.06',
                                   facecolor='#eef5f5' if i==2 else '#f1f3f5',edgecolor='#137c85' if i==2 else '#71808b',lw=1.1))
        ax.text(x+w/2,1.575,label,ha='center',va='center',fontsize=10)
        if i<3:ax.annotate('',xy=(xs[i+1]-.035,1.575),xytext=(x+w+.035,1.575),arrowprops=dict(arrowstyle='->',color='#495662',lw=1.2))
    ax.text(xs[2]+w/2,2.08,'Added in Study 2',ha='center',va='center',fontsize=9,color='#137c85')
    ax.text(0,.86,'Study 1',weight='bold',fontsize=9.5);ax.text(1.4,.86,'Show consequences at voting, action, both or neither.',fontsize=9.5)
    ax.text(0,.47,'Study 2',weight='bold',fontsize=9.5);ax.text(1.4,.47,'Add earlier history and discussion; compare with fresh context.',fontsize=9.5)
    ax.text(0,.08,'Follow-up',weight='bold',fontsize=9.5);ax.text(1.4,.08,'Keep or reset earlier context before the same final decision.',fontsize=9.5)
    export(fig,out,'protocol')
