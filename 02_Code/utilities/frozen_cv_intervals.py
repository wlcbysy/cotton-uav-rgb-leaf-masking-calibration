"""Fixed-center source-OOF residual intervals, without any model fitting.

This requested CV+ approximation is NOT standard CV+: fold-specific target
predictions are not used. No distribution-free coverage guarantee is claimed.
"""
from pathlib import Path
import hashlib,json,itertools,math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
F=ROOT/'10_Results/cotton_rgb'/(ROOT/'10_Results/cotton_rgb/CURRENT_FORMAL_VERSION.txt').read_text().strip()
OUT=F/'analysis/interval_revision'
SOURCES=[F/'analysis/raw_rgb_mining/support_predictions.csv',F/'analysis/method_strengthening/resnet50_partial_ridge_predictions.csv',F/'analysis/method_strengthening/transfer_intercept_diagnostics.csv']
DOMAINS=['2023_F1','2024_F1','2024_F2']
MODELS={'N-only':'N','N+N0':'N','N+C+T':'N_C_T','ResNet-50':'N_ResNet50_PCA52_common4'}

def compute():
    ordinary=pd.read_csv(SOURCES[0]).query("variant=='common_DAS' and algorithm=='PartialRidge'")
    deep=pd.read_csv(SOURCES[1]);offsets=pd.read_csv(SOURCES[2]).query("support=='non_N0_held_out' and model=='N_plus_N0'")
    rows=[];intervals=[];scores=[]
    for s in DOMAINS:
        for model,spec in MODELS.items():
            data=deep if model=='ResNet-50' else ordinary
            cal=data.loc[data.source.eq(s)&data.target.eq(s)&data.evaluation.eq('BlockCV')&data.specification.eq(spec)&data.N.ne(0)].sort_values('plot_id')
            assert len(cal)=={'2023_F1':21,'2024_F1':16,'2024_F2':18}[s] and cal.plot_id.is_unique
            for r in cal.itertuples():
                assert str(r.plot_id) not in str(r.train_plot_ids).split(';')
                assert r.source_block_omitted==r.block
                scores.append(dict(source=s,model=model,plot_id=r.plot_id,block=r.block,observed=r.observed,predicted=r.predicted,score=abs(r.observed-r.predicted),train_plot_ids=r.train_plot_ids,note='Unshifted source N-only OOF scores; target N0 offset changes center only' if model=='N+N0' else 'Source held-block absolute residual'))
            n=len(cal);rank=math.ceil(.9*(n+1));q=np.sort(abs(cal.observed-cal.predicted))[rank-1] if rank<=n else np.inf
            for t in DOMAINS:
                ev='BlockCV' if s==t else 'adjacent_field' if s[:4]==t[:4] else 'cross_year'
                common=dict(source=s,target=t,evaluation=ev,model=model,calibration_n=n,quantile_rank=rank,nominal_coverage=.9,support='non_N0_held_out',method='fixed_center_pooled_source_OOF',standard_CV_plus=False)
                if s==t:
                    rows.append(dict(common,target_n=n,coverage=np.nan,coverage_low=np.nan,coverage_high=np.nan,mean_width=np.nan,radius=np.nan,status='NOT_ESTIMABLE',note='Within-field remains not estimable under unchanged nested block protocol'))
                    continue
                z=data.loc[data.source.eq(s)&data.target.eq(t)&data.evaluation.eq(ev)&data.specification.eq(spec)&data.N.ne(0)].copy().sort_values('plot_id')
                assert z.plot_id.is_unique and len(z)=={'2023_F1':21,'2024_F1':16,'2024_F2':18}[t]
                offset=float(offsets.loc[offsets.source.eq(s)&offsets.target.eq(t),'N0_intercept'].iloc[0]) if model=='N+N0' else 0.
                z['center']=z.predicted+offset;z['lower']=z.center-q;z['upper']=z.center+q
                z['covered']=(z.observed>=z.lower)&(z.observed<=z.upper)
                boot=[]
                for blocks in itertools.product([1,2,3],repeat=3):
                    ix=np.concatenate([np.flatnonzero(z.block.to_numpy()==b) for b in blocks]);boot.append(z.covered.iloc[ix].mean())
                low,high=np.quantile(boot,[.025,.975])
                rows.append(dict(common,target_n=len(z),coverage=z.covered.mean(),coverage_low=low,coverage_high=high,mean_width=2*q,radius=q,status='FINITE_EMPIRICAL',note='No source- or target-domain guarantee for this fixed-center approximation; only 3 target blocks'))
                for r in z.itertuples():intervals.append(dict(common,plot_id=r.plot_id,block=r.block,N=r.N,observed=r.observed,frozen_prediction=r.predicted,stored_N0_offset=offset,center=r.center,lower=r.lower,upper=r.upper,covered=r.covered))
    return pd.DataFrame(rows),pd.DataFrame(intervals),pd.DataFrame(scores)

def main():
    OUT.mkdir(exist_ok=True)
    summary,intervals,scores=compute()
    for name,d in [('summary',summary),('intervals',intervals),('calibration_scores',scores)]:d.to_csv(OUT/f'frozen_center_cv_{name}.csv',index=False)
    protocol=dict(method='Fixed-center pooled source out-of-fold residual intervals (requested CV+ approximation)',formula='q = score_(ceil(0.90*(n+1))); interval = frozen target prediction + stored N0 offset if applicable +/- q',standard_CV_plus=False,reason='Standard CV+ aggregates fold-specific target predictions +/- paired residuals; frozen full-source centers omit that variation.',guarantee='CV+ theory requires exchangeability and further fold assumptions; it does not transfer automatically across fields/years. This fixed-center approximation has no standard CV+ finite-sample guarantee even in source domain.',N0='Use source N-only OOF residuals unchanged; use the already stored target three-N0 shift for the target center; no target holdout labels used in construction.',uncertainty='27 ordered resamples of three target blocks, conditional on frozen intervals; descriptive pointwise 95% percentile limits, not reliable new-domain uncertainty.',sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES})
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(summary.loc[summary.source.ne(summary.target),['source','target','model','calibration_n','quantile_rank','coverage','mean_width']].to_string(index=False))
if __name__=='__main__':main()
