"""Paper-wide visual vocabulary; all sizes are final-print points/millimetres."""
from pathlib import Path
import matplotlib as mpl
import numpy as np
import re
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory
MM = 1 / 25.4
DOUBLE_MM, SINGLE_MM = 180, 90
ENV_COLORS = {'2023_F1':'#B45309','2024_F1':'#0284C7','2024_F2':'#0F766E'}
ENV_MARKERS = {'2023_F1':'s','2024_F1':'o','2024_F2':'^'}
ENV_LABELS = {k:k.replace('_',' ') for k in ENV_COLORS}
MODEL_COLORS = {'N':'#475569','N_Cearly':'#B45309','N_C':'#64748B','N_C_T':'#0284C7','N_Leaf_T':'#0F766E','N_B':'#882255','N_C_B':'#882255','N_C_T_B':'#882255','RGB52':'#334155','ResNet50':'#9F1239'}
ERROR_COLORS = {'W_err':'#0F766E','B_err':'#D97706','Bias2':'#64748B'}
N_LEVELS = [0,75,150,225,300,375,450]
N_COLORS = {0:'#B45309',75:'#93C5FD',150:'#38BDF8',225:'#0284C7',300:'#0D9488',375:'#0F766E',450:'#134E4A'}
BLOCK_FILL = {1:'full',2:'left',3:'none'}
ROLES = [('2023_F1','2023_F1','BlockCV'),('2024_F1','2024_F1','BlockCV'),('2024_F2','2024_F2','BlockCV'),('2024_F1','2024_F2','adjacent_field'),('2024_F2','2024_F1','adjacent_field'),('2023_F1','2024_F1','cross_year'),('2023_F1','2024_F2','cross_year'),('2024_F1','2023_F1','cross_year'),('2024_F2','2023_F1','cross_year')]
TASK_LABELS = {(s,t,e):ENV_LABELS[s]+' CV' if e=='BlockCV' else ENV_LABELS[s]+' → '+ENV_LABELS[t] for s,t,e in ROLES}
GROUPS = [('Within-field BlockCV',0,2),('Adjacent-field',3,4),('Cross-year',5,8)]

MODEL_LABELS = {'N':r'$N$-only','N_Cearly':r'$N+C_{\mathrm{early}}$','N_C':r'$N+C$','N_C_T':r'$N+C+T$','N_B':r'$N+B$ vs. $N$','N_C_B':r'$N+C+B$ vs. $N+C$','N_C_T_B':r'$N+C+T+B$ vs. $N+C+T$','RGB52':r'$N+\mathrm{RGB52}$'}
ERROR_LABELS = {'W_err':r'$W_{\mathrm{err}}$','B_err':r'$B_{\mathrm{err}}$','Bias2':r'Bias$^2$'}

def scientific_text(text):
    """Format variables outside existing math spans; data column names stay untouched."""
    parts=re.split(r'(\$[^$]*\$)',text)
    for i in range(0,len(parts),2):
        v=parts[i]
        for old,new in [('C_early',r'$C_{\mathrm{early}}$'),('T_auc_mean',r'$T_{\mathrm{auc,mean}}$'),('W_err',r'$W_{\mathrm{err}}$'),('B_err',r'$B_{\mathrm{err}}$')]:v=v.replace(old,new)
        parts[i]=v
    return ''.join(parts)

def apply():
    import matplotlib._mathtext as mt
    mt.SHRINK_FACTOR = 1.0
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
        'font.size':7,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'legend.title_fontsize':7,'mathtext.fontset':'custom','mathtext.rm':'Arial',
        'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.6,'legend.frameon':False,'savefig.facecolor':'white'})

def normalize(fig):
    """Final safety net for legacy explicit typography, without scaling text down."""
    if abs(fig.get_figwidth()/MM-DOUBLE_MM) > .01:
        fig.set_size_inches(DOUBLE_MM*MM, fig.get_figheight(), forward=True)
    for t in fig.findobj(Text):
        s=t.get_text().strip()
        t.set_text(scientific_text(t.get_text()))
        t.set_fontfamily('Arial')
        t.set_fontsize(9 if len(s)==1 and s.isalpha() and t.get_fontweight()=='bold' else max(7,t.get_fontsize()))
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
        ax.tick_params(labelsize=7)
        for leg in [ax.get_legend()]:
            if leg:
                for t in leg.get_texts():t.set_fontsize(7)
    for leg in fig.legends:
        for t in leg.get_texts():t.set_fontsize(7)

def save(fig, directory, stem):
    normalize(fig); directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png','tiff'):
        kw={'dpi':600 if ext=='tiff' else 240} if ext in ('png','tiff') else {}
        if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(directory/f'{stem}.{ext}',bbox_inches=None,**kw)

def task_axis(ax, labels=True, zero_line: bool = True):
    pos=np.array([0,1,2,3.8,4.8,6.6,7.6,8.6,9.6])
    ax.set_yticks(pos,[TASK_LABELS[r] for r in ROLES] if labels else ['']*9)
    ax.set_ylim(10.15,-1.0)
    if labels:
        for name,a,b in GROUPS:
            ax.text(.012,pos[a]-.65,name,transform=blended_transform_factory(ax.figure.transFigure,ax.transData),ha='left',va='center',fontsize=7,fontweight='bold')
    if zero_line: ax.axvline(0,color='#777777',lw=.6,ls=':')
    return pos
