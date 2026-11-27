"""Management-matched yield evaluation of cotton proxies and canopy trajectories.

All registered feature specifications and directions are reported. No target
yield enters fitting, scaling, or tuning. Existing empirical inputs stay locked.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr,spearmanr

ROOT=Path(__file__).resolve().parents[3]
F=ROOT/'10_Results/cotton_rgb'/(ROOT/'10_Results/cotton_rgb/CURRENT_FORMAL_VERSION.txt').read_text().strip()
OUT=F/'analysis/raw_rgb_mining'
INDICES=['ExB','ExG','ExR','ExGR','GBRI','GLI','IKAW','MGRVI','NGBDI','NGRDI','RGBVI','RGRI','VARI']
ALPHAS=[0.1,1,10,100,1000,10000,100000]
BASE_SPECS=['N','C','B','C_B','N_C','N_B','N_C_B','N_C_T','N_C_T_B','N_Cearly','N_RGB13_S6','N_RGB39']
TUNE=[];PRED=[]

class PartialRidge:
    """Joint fit with unpenalized [1,N,N²] and train-scaled optical terms."""
    def __init__(self,alpha):self.alpha=alpha
    def fit(self,x,y):
        a=np.c_[np.ones(len(x)),x[:,0]/450,x[:,1]/450**2]
        self.base=np.linalg.lstsq(a,y,rcond=None)[0]
        self.n_optical=x.shape[1]-2
        if self.n_optical:
            self.scale=StandardScaler().fit(x[:,2:]);z=self.scale.transform(x[:,2:])
            self.proj=np.linalg.lstsq(a,z,rcond=None)[0];rz=z-a@self.proj;ry=y-a@self.base
            self.gamma=np.linalg.solve(rz.T@rz+self.alpha*np.eye(rz.shape[1]),rz.T@ry)
        return self
    def predict(self,x):
        a=np.c_[np.ones(len(x)),x[:,0]/450,x[:,1]/450**2];pred=a@self.base
        if self.n_optical:pred+=(self.scale.transform(x[:,2:])-a@self.proj)@self.gamma
        return pred

def build_table(direct,legacy,white,green='green_canopeo'):
    rows=[]
    for (yr,field,plot),d in direct.groupby(['year','field','plot_id'],sort=True):
        d=d.sort_values('DAS');end=142 if yr==2023 else 141;early=130 if yr==2023 else 128
        s6=d[d.DAS==181].iloc[0];late=d[d.DAS==end].iloc[0];e=d[d.DAS==early].iloc[0]
        # Only direct data <= cutoff: interpolation at 92 may use prior direct
        # dates and the next DIRECT date, which must itself be <= the cutoff.
        history=d[d.DAS<=end];xs=history.DAS.to_numpy(float);ys=history[green].to_numpy(float)
        xx=np.r_[92.,xs[(xs>92)&(xs<end)],float(end)]
        yy=np.interp(xx,xs,ys)
        assert xs.min()<=92 and xs.max()>=end
        row={k:s6[k] for k in ['year','field','plot_id','block_design','N_kg_ha','yield_kg_ha']}
        row.update(domain=f'{yr}_{field}',C_cover=late[green],C_exg=late.exg_all_valid_mean,
                   Cearly_cover=e[green],Cearly_exg=e.exg_all_valid_mean,
                   B_white=s6[white],T_auc_mean=np.trapezoid(yy,xx)/(end-92),
                   T_late_change=(late[green]-e[green])/(end-early)*(end-92),
                   T_direct_dates=int(((history.DAS>=92)&(history.DAS<=end)).sum()),
                   T_start_interpolated=not bool((history.DAS==92).any()),
                   C_cutoff=end,Cearly_cutoff=early,B_cutoff=181,
                   B_white_spatial=s6.get(white+'_spatial_sd',np.nan),
                   C_spatial=late[green+'_spatial_sd'] if green+'_spatial_sd' in late.index else np.nan)
        raw=legacy[(legacy.year==yr)&(legacy.field==field)&(legacy.plot_id==plot)]
        for name,ds in [('S5a',early),('S5b',end),('S6',181)]:
            z=raw[raw.DAS==ds].iloc[0]
            assert z.slot_status=='AVAILABLE'
            row.update({f'{name}_{col}':z[col] for col in INDICES})
        rows.append(row)
    result=pd.DataFrame(rows)
    assert len(result)==66 and not result.duplicated(['domain','plot_id']).any()
    return result

def design(d,spec):
    n=d.N_kg_ha.to_numpy(float);parts=[]
    for token in spec.split('_'):
        if token=='N':parts.append(np.c_[n,n*n])
        elif token=='C':parts.append(d[['C_cover','C_exg']].to_numpy(float))
        elif token=='Cearly':parts.append(d[['Cearly_cover','Cearly_exg']].to_numpy(float))
        elif token=='B':parts.append(d[['B_white']].to_numpy(float))
        elif token=='T':parts.append(d[['T_auc_mean','T_late_change']].to_numpy(float))
        elif token=='RGB13':parts.append(d[[f'S6_{c}' for c in INDICES]].to_numpy(float))
        elif token=='S6':pass
        elif token=='RGB39':parts.append(d[[f'{s}_{c}' for s in ['S5a','S5b','S6'] for c in INDICES]].to_numpy(float))
        else:raise ValueError(token)
    x=np.column_stack(parts);assert np.isfinite(x).all();return x

def model(family,param=None):
    if family=='OLS':return make_pipeline(StandardScaler(),LinearRegression())
    if family=='PartialRidge':return PartialRidge(param)
    if family=='Ridge':reg=make_pipeline(StandardScaler(),Ridge(alpha=param))
    else:reg=RandomForestRegressor(n_estimators=100,max_depth=3,min_samples_leaf=3,max_features=1.0,n_jobs=1,random_state=20261025)
    return TransformedTargetRegressor(regressor=reg,transformer=StandardScaler())

def fit(d,family,spec,train,variant,role):
    x=design(d,spec);y=d.yield_kg_ha.to_numpy(float);g=d.block_design.to_numpy();alpha=None;inner=np.nan
    if family in ['Ridge','PartialRidge']:
        rank=[]
        for a in ALPHAS:
            pred=np.full(len(train),np.nan)
            for block in np.unique(g[train]):
                tr=train[g[train]!=block];te=train[g[train]==block]
                pred[g[train]==block]=model(family,a).fit(x[tr],y[tr]).predict(x[te])
            err=pred-y[train];rank.append((float(np.sqrt(np.mean(err**2))),float(np.mean(abs(err))),-a,a))
        inner,mae,_,alpha=min(rank)
    m=model(family,alpha).fit(x[train],y[train])
    TUNE.append({'variant':variant,'source':str(d.domain.iloc[0]),'algorithm':family,'specification':spec,'role':role,'train_n':len(train),'alpha':alpha,'inner_RMSE':inner})
    return m

def add(m,src,target,fam,spec,role,variant,idx=None):
    dd=target if idx is None else target.iloc[idx]
    yh=m.predict(design(dd,spec))
    for (_,r),p in zip(dd.iterrows(),yh):
        PRED.append({'variant':variant,'source':src,'target':r.domain,'algorithm':fam,'specification':spec,'evaluation':role,'plot_id':int(r.plot_id),'block':int(r.block_design),'N':float(r.N_kg_ha),'observed':float(r.yield_kg_ha),'predicted':float(p)})

def run_variant(table,variant,specs,source_sensitivity=False,families=None):
    sets={k:d.reset_index(drop=True) for k,d in table.groupby('domain')}
    for src,d in sets.items():
        ii=np.arange(len(d));groups=d.block_design.to_numpy()
        for family in (families or ['Ridge','RF','OLS']):
            use=specs if family!='OLS' else ['N']
            for spec in use:
                for b in np.unique(groups):
                    tr=ii[groups!=b];te=ii[groups==b]
                    m=fit(d,family,spec,tr,variant,f'hold_block_{b}')
                    add(m,src,d,family,spec,'BlockCV',variant,te)
                    if source_sensitivity and src=='2024_F1' and spec in ['N','N_C','N_B','N_C_B','N_C_T_B']:
                        add(m,src,sets['2024_F2'],family,spec,f'source_omit_block_{b}',variant)
                m=fit(d,family,spec,ii,variant,'full_source')
                for tar,td in sets.items():
                    if tar==src:continue
                    role='adjacent_field' if src[:4]==tar[:4] else 'cross_year'
                    add(m,src,td,family,spec,role,variant)
        print('MODELS',variant,src,len(PRED),'predictions',flush=True)

def summarize(pred):
    keys=['variant','source','target','algorithm','specification','evaluation'];rows=[];nrows=[]
    for key,d in pred.groupby(keys,sort=False):
        e=d.predicted-d.observed;bias=e.mean();en=e.groupby(d.N).transform('mean');b=np.mean((en-bias)**2);w=np.mean((e-en)**2);mse=np.mean(e**2)
        assert np.isclose(mse,bias*bias+b+w,atol=1e-6,rtol=1e-12)
        r={**dict(zip(keys,key)),'n':len(d),'RMSE':np.sqrt(mse),'MAE':np.mean(abs(e)),'R2':r2_score(d.observed,d.predicted),'Bias':bias,'cRMSE':np.std(e),'Bias2':bias*bias,'BetweenN':b,'WithinN':w,'within_N_RMSE':np.sqrt(w),'min_prediction':d.predicted.min(),'max_prediction':d.predicted.max()}
        rows.append(r)
        for n,dd in d.groupby('N'):
            err=dd.predicted-dd.observed
            nrows.append({**dict(zip(keys,key)),'N':n,'n':len(dd),'observed_mean':dd.observed.mean(),'predicted_mean':dd.predicted.mean(),'mean_error':err.mean(),'within_N_MSE':np.mean((err-err.mean())**2)})
    met=pd.DataFrame(rows)
    for ix,r in met.iterrows():
        same=(met.variant==r.variant)&(met.source==r.source)&(met.target==r.target)&(met.evaluation==r.evaluation)&(met.specification=='N')
        for prefix,alg in [('matched',r.algorithm),('OLS','OLS')]:
            base=met[same&(met.algorithm==alg)]
            if len(base)==1:
                for col in ['RMSE','MAE','within_N_RMSE']:
                    met.loc[ix,f'gain_{prefix}_{col}']=float(base.iloc[0][col]-r[col])
    return met,pd.DataFrame(nrows)

def associations(table):
    out=[]
    for domain,d in table.groupby('domain'):
        z=pd.get_dummies(d[['N_kg_ha','block_design']].astype(str),drop_first=True,dtype=float)
        z=np.column_stack([np.ones(len(d)),z.to_numpy()])
        y=d.yield_kg_ha.to_numpy(float);ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
        for col in ['C_cover','C_exg','B_white','T_auc_mean','T_late_change']:
            x=d[col].to_numpy(float);rx=x-z@np.linalg.lstsq(z,x,rcond=None)[0]
            out.append({'domain':domain,'feature':col,'n':len(d),'Pearson_r':pearsonr(x,y).statistic,'Spearman_r':spearmanr(x,y).statistic,'N_block_adjusted_r':pearsonr(rx,ry).statistic,'feature_min':min(x),'feature_max':max(x),'feature_mean':np.mean(x),'feature_sd':np.std(x)})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial-only',action='store_true');args=ap.parse_args()
    if args.partial_only:
        global PRED,TUNE
        prior=pd.read_csv(OUT/'predictions.csv');PRED=prior[prior.algorithm!='PartialRidge'].to_dict('records')
        prior=pd.read_csv(OUT/'tuning.csv');TUNE=prior[prior.algorithm!='PartialRidge'].to_dict('records')
    direct=pd.read_csv(OUT/'direct_image_features.csv');legacy=pd.read_csv(ROOT/'01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv')
    sel=json.loads((OUT/'mask_selection.json').read_text());white=sel['cotton_rule']
    table=build_table(direct,legacy,white);table.to_csv(OUT/'plot_features.csv',index=False,float_format='%.12g')
    associations(table).to_csv(OUT/'feature_associations.csv',index=False,float_format='%.12g')
    main_specs=[s for s in BASE_SPECS if s.startswith('N')] if args.partial_only else BASE_SPECS
    families=['PartialRidge'] if args.partial_only else None
    run_variant(table,'primary',main_specs,True,families)
    brightness=int(white.split('_')[1][1:]);sat=white.split('_')[2]
    specs=['N','N_C','N_B','N_C_B','N_C_T','N_C_T_B']
    for t in [brightness-20,brightness+20]:
        rule=f'white_b{t}_{sat}'
        if rule not in direct:continue
        tab=build_table(direct,legacy,rule);run_variant(tab,f'white_b{t}',specs,families=families)
    for green in ['green_loose','green_strict']:
        tab=build_table(direct,legacy,white,green);run_variant(tab,green,specs,families=families)
    pred=pd.DataFrame(PRED);pred.to_csv(OUT/'predictions.csv',index=False,float_format='%.12g')
    met,within=summarize(pred);met.to_csv(OUT/'model_metrics.csv',index=False,float_format='%.12g')
    within.to_csv(OUT/'within_N_metrics.csv',index=False,float_format='%.12g')
    pd.DataFrame(TUNE).to_csv(OUT/'tuning.csv',index=False,float_format='%.12g')
    receipt={'status':'PASS','plot_year_records':len(table),'direct_image_slots':len(direct),'prediction_rows':len(pred),'metric_rows':len(met),'source_fits':len(TUNE),'all_predictions_finite':bool(np.isfinite(pred.predicted).all()),'decomposition_identity':'PASS','target_yield_used_in_fitting_or_tuning':False,'selection':'image mask chosen using F1 RGB reference points; model tuning source blocks only; all specifications reported','harvest_confirmation':'both years no picking before October13 flight; confirmed','note':'Exploratory retrospective evaluation with point-grid visual verification.'}
    (OUT/'analysis_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
