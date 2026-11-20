"""Additive exploratory analysis; never write into frozen analysis directories."""
from pathlib import Path
from itertools import product
import sys, json, argparse, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit
from threadpoolctl import threadpool_limits
import mine_raw_rgb_information as legacy

ROOT=legacy.ROOT; FORMAL=legacy.F; OLD=legacy.OUT
OUT=FORMAL/'analysis/data_mining_current'
OUT.mkdir(exist_ok=True)
COMMON=pd.read_csv(OLD/'common_plot_features_common_DAS.csv')
DOMAINS=sorted(COMMON.domain.unique())
DIRECT=pd.read_csv(OLD/'direct_image_features.csv')
DIRECT['domain']=DIRECT.year.astype(str)+'_'+DIRECT.field
TASKS=[(s,t,'BlockCV' if s==t else 'adjacent_field' if s[:4]==t[:4] else 'cross_year') for s in DOMAINS for t in DOMAINS]
TUNING=[]

def save(d,name):
    pd.DataFrame(d).to_csv(OUT/(name+'.csv'),index=False)

def clocks():
    weather=pd.read_csv(ROOT/'01_Data/processed/weather/alar_2023_2024_daily_weather.csv')
    weather['date']=pd.to_datetime(weather.date)
    rows=[]
    for dom in DOMAINS:
        sow=pd.Timestamp(dom[:4]+'-04-15')
        d=weather.loc[weather.date.between(sow,pd.Timestamp(dom[:4]+'-10-13'))].copy()
        assert len(d)==182 and d.T_mean.notna().all() and d.date.diff().dropna().eq(pd.Timedelta(days=1)).all()
        d['daily_GDD']=np.maximum(d.T_mean-10,0);d['GDD']=d.daily_GDD.cumsum()
        d['DAS']=(d.date-sow).dt.days;d['domain']=dom;d['base_C']=10;rows.append(d)
    daily=pd.concat(rows);save(daily,'daily_thermal_time')
    lookup=daily.set_index(['domain','date']).GDD
    DIRECT['GDD']=[lookup.loc[(r.domain,pd.Timestamp(r.flight_date))] for r in DIRECT.itertuples()]
    flights=DIRECT[['domain','flight_date','DAS','GDD']].drop_duplicates().sort_values(['domain','DAS'])
    save(flights,'flight_thermal_time');return daily,flights

def fit(x,y,blocks,train,trace):
    """Identical alpha grid, source-only scaling and inner leave-block-out ranking."""
    assert len(np.unique(blocks[train]))>=2
    scores=[]
    for alpha in legacy.ALPHAS:
        pred=np.empty(len(train))
        for b in np.unique(blocks[train]):
            keep=blocks[train]!=b;tr=train[keep];te=train[~keep]
            model=legacy.PartialRidge(alpha).fit(x[tr],y[tr]);pred[~keep]=model.predict(x[te])
        scores.append((np.sqrt(np.mean((pred-y[train])**2)),np.mean(abs(pred-y[train])),-alpha))
    best=-min(scores)[2];TUNING.append(dict(trace,alpha=best,train_n=len(train),train_indices=';'.join(map(str,train))))
    return legacy.PartialRidge(best).fit(x[train],y[train])

def evaluate(frames,feature_cols,source,target,variant,meta):
    a=frames[source].reset_index(drop=True);b=frames[target].reset_index(drop=True)
    y=a.yield_kg_ha.to_numpy();groups=a.block_design.to_numpy();ii=np.arange(len(a));rows=[]
    folds=np.unique(groups) if source==target else [0]
    for fold in folds:
        tr=ii[groups!=fold] if fold else ii
        te=np.flatnonzero(b.block_design.eq(fold)) if fold else np.arange(len(b))
        def design(d):
            n=d.N_kg_ha.to_numpy();return np.c_[n,n*n,d[feature_cols].to_numpy()] if feature_cols else np.c_[n,n*n]
        xa,xb=design(a),design(b)
        # Non-finite features stop the comparison; support is not changed or imputed.
        assert np.isfinite(xa).all() and np.isfinite(xb).all()
        model=fit(xa,y,groups,tr,dict(meta,variant=variant,source=source,target=target,held_block=int(fold)))
        pred=model.predict(xb[te])
        for r,p in zip(b.iloc[te].itertuples(),pred):
            rows.append(dict(meta,variant=variant,source=source,target=target,evaluation='BlockCV' if source==target else 'adjacent_field' if source[:4]==target[:4] else 'cross_year',plot_id=r.plot_id,block=r.block_design,N=r.N_kg_ha,observed=r.yield_kg_ha,predicted=p,train_plot_ids=';'.join(map(str,a.iloc[tr].plot_id)),held_block=int(fold)))
    return rows

def paired_summary(pred,group):
    out=[]
    for key,d in pred.groupby(group,dropna=False,sort=False):
        base=d.loc[d.variant.eq('N')].sort_values('plot_id')
        for model,z in d.groupby('variant'):
            if model=='N':continue
            z=z.sort_values('plot_id');assert z.plot_id.tolist()==base.plot_id.tolist()
            a=base.predicted.to_numpy()-base.observed.to_numpy();b=z.predicted.to_numpy()-z.observed.to_numpy()
            blocks=z.block.to_numpy();members=[np.flatnonzero(blocks==v) for v in np.unique(blocks)]
            vals=[]
            for picks in product(members,repeat=3):
                ix=np.concatenate(picks);vals.append(np.sqrt(np.mean(a[ix]**2))-np.sqrt(np.mean(b[ix]**2)))
            out.append(dict(zip(group,key),variant=model,n=len(z),RMSE_N=np.sqrt(np.mean(a*a)),RMSE_model=np.sqrt(np.mean(b*b)),gain=np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b)),low=np.quantile(vals,.025),high=np.quantile(vals,.975),resamples=27))
    return pd.DataFrame(out)

def history_features(dom,clock,cut):
    base=COMMON.loc[COMMON.domain.eq(dom)].copy();rows=[]
    for r in base.itertuples():
        d=DIRECT.loc[DIRECT.domain.eq(dom)&DIRECT.plot_id.eq(r.plot_id)&DIRECT[clock].le(cut)].sort_values(clock)
        if d.empty:return None
        x=d[clock].to_numpy();c=d.green_canopeo.to_numpy();last=d.iloc[-1]
        span=x[-1]-x[0]
        rows.append(dict(plot_id=r.plot_id,H_cover=c[-1],H_exg=last.exg_all_valid_mean,
                         H_auc=np.trapezoid(c,x)/span if span else c[-1],
                         H_change=(c[-1]-c[-2])/(x[-1]-x[-2])*span if len(d)>1 else 0,
                         first_DAS=d.DAS.iloc[0],last_DAS=last.DAS,last_GDD=last.GDD,n_flights=len(d),last_date=last.flight_date))
    return base.merge(pd.DataFrame(rows),on='plot_id',validate='one_to_one')

def timing():
    _,flights=clocks();pred=[];trace=[]
    for source,target,ev in TASKS:
        for clock in ['DAS','GDD']:
            for flight in flights.loc[flights.domain.eq(target)].itertuples():
                cut=getattr(flight,clock);a=history_features(source,clock,cut);b=history_features(target,clock,cut)
                if a is None or b is None:continue
                frames={source:a,target:b};meta=dict(clock=clock,cutoff=cut,target_date=flight.flight_date,target_DAS=flight.DAS,target_GDD=flight.GDD)
                for dom,v in frames.items():
                    trace.extend(v[['plot_id','first_DAS','last_DAS','last_GDD','n_flights','last_date']].assign(source=source,target=target,domain=dom,**meta).to_dict('records'))
                for variant,cols in [('N',[]),('C',['H_cover','H_exg']),('C+T',['H_cover','H_exg','H_auc','H_change'])]:
                    pred.extend(evaluate(frames,cols,source,target,variant,meta))
        print('TIMING',source,target,flush=True)
    pred=pd.DataFrame(pred);save(pred,'recursive_predictions');save(trace,'recursive_feature_provenance')
    metrics=paired_summary(pred,['source','target','evaluation','clock','cutoff','target_date','target_DAS','target_GDD']);save(metrics,'recursive_metrics')
    stable=[]
    for key,d in metrics.groupby(['source','target','evaluation','clock','variant']):
        d=d.sort_values('cutoff');good=d.low.gt(0).to_numpy();found=None
        # At least three consecutive acquisitions and no later reversal; descriptive, not a change-point test.
        for i in range(len(d)-2):
            if good[i:].all():found=d.iloc[i];break
        meaningful=(d.low.gt(0)&d.gain.ge(15)).to_numpy();practical=None
        for i in range(len(d)-2):
            if meaningful[i:].all():practical=d.iloc[i];break
        stable.append(dict(zip(['source','target','evaluation','clock','variant'],key),stable=found is not None,first_date=found.target_date if found is not None else '',first_DAS=found.target_DAS if found is not None else np.nan,first_GDD=found.target_GDD if found is not None else np.nan,available_steps=len(d),practical_stable=practical is not None,practical_first_date=practical.target_date if practical is not None else ''))
    save(stable,'earliest_stable_dates');save(TUNING,'recursive_tuning')

def single(t,a,k,mid):return a*expit(k*(t-mid))
def double(t,a,k1,m1,k2,m2):return a*expit(k1*(t-m1))*expit(-k2*(t-m2))

def growth():
    daily,_=clocks();rows=[]
    for (dom,plot),d in DIRECT.groupby(['domain','plot_id']):
        d=d.sort_values('DAS');x=d.DAS.to_numpy(float);y=d.green_canopeo.to_numpy(float);candidates=[]
        for name,fun,bounds,starts in [
            ('logistic',single,([0,-1,0],[1,1,230]),[[.95,.05,50],[.95,-.05,160]]),
            ('double_logistic',double,([0,.002,0,.002,80],[1,1,140,1,260]),[[.98,.05,50,.04,170],[.98,.1,70,.1,150]])]:
            for start in starts:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');p,cov=curve_fit(fun,x,y,p0=start,bounds=bounds,maxfev=20000)
                    sse=float(np.sum((fun(x,*p)-y)**2));k=len(p);aicc=len(x)*np.log(max(sse/len(x),1e-15))+2*k+2*k*(k+1)/(len(x)-k-1)
                    at_bound=bool(np.any(np.isclose(p,bounds[0],atol=1e-4)) or np.any(np.isclose(p,bounds[1],atol=1e-4)))
                    candidates.append((aicc,name,p,fun,sse,at_bound))
                except (ValueError,RuntimeError,FloatingPointError):pass
        if not candidates:
            rows.append(dict(domain=dom,plot_id=plot,status='FAILED',n_dates=len(x)));continue
        best=min(candidates,key=lambda z:z[0]);score,name,p,fun,sse,bound=best
        grid=np.linspace(x[0],x[-1],1001);curve=fun(grid,*p);peak=float(grid[np.argmax(curve)])
        derivative=np.gradient(curve,grid);thermal=daily.loc[daily.domain.eq(dom)]
        rows.append(dict(domain=dom,plot_id=plot,status='REVIEW_BOUNDARY' if bound else 'FIT',n_dates=len(x),model=name,AICc=score,fit_RMSE=np.sqrt(sse/len(x)),parameters=json.dumps(p.tolist()),peak=float(curve.max()),peak_DAS=peak,peak_GDD=float(np.interp(peak,thermal.DAS,thermal.GDD)),early_slope=float(np.max(derivative[grid<=np.quantile(grid,.33)])),late_decline=float(-np.mean(derivative[grid>=np.quantile(grid,.8)])),peak_boundary=bool(np.argmax(curve) in (0,1000)),observed_start_DAS=x[0],observed_end_DAS=x[-1]))
    curves=pd.DataFrame(rows);save(curves,'growth_curve_features')
    save(curves.loc[curves.status.ne('FIT')|curves.peak_boundary.fillna(True)],'growth_fit_review_flags')
    cols=['peak','peak_DAS','peak_GDD','early_slope','late_decline']
    table=COMMON.merge(curves,on=['domain','plot_id'],validate='one_to_one')
    # No silent imputation or complete-case change of the frozen 64-plot evaluation support.
    assert table[cols].notna().all().all(),'Growth failure: comparison paused; inspect failure CSV'
    frames={dom:d for dom,d in table.groupby('domain')};pred=[]
    for s,t,_ in TASKS:
        for v,c in [('N',[]),('C',['C_cover','C_exg']),('C+T',['C_cover','C_exg','T_auc_mean','T_late_change']),('Growth',cols)]:
            pred.extend(evaluate(frames,c,s,t,v,{}))
    pred=pd.DataFrame(pred);save(pred,'growth_predictions');save(paired_summary(pred,['source','target','evaluation']),'growth_comparison');save(TUNING,'growth_tuning')
    frozen=pd.read_csv(OLD/'support_predictions.csv').query("variant=='common_DAS' and algorithm=='PartialRidge' and specification in ['N','N_C','N_C_T'] and evaluation in ['BlockCV','adjacent_field','cross_year']")
    frozen['variant']=frozen.specification.map({'N':'N','N_C':'C','N_C_T':'C+T'})
    z=pred.merge(frozen,on=['source','target','evaluation','variant','plot_id'],suffixes=('_new','_frozen'),validate='one_to_one')
    error=float(abs(z.predicted_new-z.predicted_frozen).max());assert len(z)==576 and error<1e-5
    (OUT/'protocol_replay.json').write_text(json.dumps({'frozen_prediction_rows':len(z),'max_absolute_difference':error,'status':'PASS'},indent=2)+'\n')

def conformal():
    """Strict block split, finite-sample +infinity quantile; no fabricated finite widths."""
    frozen=pd.read_csv(OLD/'support_predictions.csv').query("variant=='common_DAS' and algorithm=='PartialRidge'")
    deep=pd.read_csv(FORMAL/'analysis/method_strengthening/resnet50_partial_ridge_predictions.csv')
    rows=[];scores=[];intervals=[]
    models={'N-only':('N',frozen),'N+N0':('N',frozen),'N+C+T':('N_C_T',frozen),'ResNet-50':('N_ResNet50_PCA52_common4',deep)}
    for s,t,ev in TASKS:
        target=COMMON.loc[COMMON.domain.eq(t)&COMMON.N_kg_ha.ne(0)];source=COMMON.loc[COMMON.domain.eq(s)]
        for model,(spec,data) in models.items():
            for block in [1,2,3]:
                cal=source.loc[source.block_design.eq(block)&source.N_kg_ha.ne(0)]
                n=len(cal);rank=int(np.ceil((n+1)*.9));assert rank>n
                # Frozen source held-block predictions are exactly the source-only fitted models,
                # with training/tuning excluding this calibration block.
                z=data.loc[data.source.eq(s)&data.target.eq(s)&data.evaluation.eq('BlockCV')&data.specification.eq(spec)&data.plot_id.isin(cal.plot_id)]
                assert len(z)==n
                for r in z.itertuples():scores.append(dict(source=s,model=model,calibration_block=block,plot_id=r.plot_id,raw_absolute_residual=abs(r.predicted-r.observed),note='N+N0 scores are unshifted N scores; offset immaterial for +infinity quantile'))
                estimable=s!=t
                for r in target.itertuples():
                    intervals.append(dict(source=s,target=t,evaluation=ev,model=model,calibration_block=block,plot_id=r.plot_id,target_block=r.block_design,N=r.N_kg_ha,observed=r.yield_kg_ha,lower=-np.inf if estimable else np.nan,upper=np.inf if estimable else np.nan,covered=True if estimable else None,status='VACUOUS_INFINITY' if estimable else 'NOT_ESTIMABLE_NESTED_BLOCK_SPLIT'))
                rows.append(dict(source=s,target=t,evaluation=ev,model=model,calibration_block=block,calibration_n=n,quantile_rank=rank,target_n=len(target),coverage=1.0 if estimable else np.nan,mean_width=np.inf if estimable else np.nan,status='VACUOUS_INFINITY' if estimable else 'NOT_ESTIMABLE_NESTED_BLOCK_SPLIT',reason='rank exceeds calibration n; no finite 90% split-conformal radius' if estimable else 'outer test plus independent calibration leaves one training block; original inner block tuning impossible',support='non_N0_held_out',target_N0_labels=3 if model=='N+N0' else 0,exchangeability_guarantee=False))
    save(pd.DataFrame(scores).drop_duplicates(),'conformal_calibration_scores');save(rows,'conformal_split_audit');save(intervals,'conformal_intervals')
    d=pd.DataFrame(rows);summary=d.groupby(['source','target','evaluation','model'],sort=False).agg(coverage=('coverage','first'),mean_width=('mean_width','first'),target_n=('target_n','first'),calibration_n_min=('calibration_n','min'),calibration_n_max=('calibration_n','max'),status=('status','first')).reset_index()
    # Infinite intervals cover every finite outcome by construction; ordinary binomial
    # uncertainty would misleadingly present this tautology as evidence of calibration.
    summary['coverage_uncertainty']=np.where(summary.status.eq('VACUOUS_INFINITY'),'Vacuous 100%; no useful finite-width coverage estimate. Only three target blocks.','Not estimable under unchanged nested block protocol; no coverage estimate.')
    save(summary,'conformal_summary')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['timing','growth','conformal']);args=ap.parse_args()
    with threadpool_limits(1):globals()[args.action]()
