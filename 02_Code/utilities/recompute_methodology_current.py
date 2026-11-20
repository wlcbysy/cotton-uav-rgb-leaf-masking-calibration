"""Current, design-matched repairs for the formal cotton RGB manuscript.

Reads protected observations; overwrites only derived current analysis outputs.
"""
from pathlib import Path
import sys, json, hashlib, itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import block_diag
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'02_Code/src/modeling'))
sys.path.insert(0, str(ROOT/'02_Code/src/data'))
import mine_raw_rgb_information as m
P = m.OUT
OUT = P/'methodology_current'
OUT.mkdir(exist_ok=True)

def partial_result(d):
    x = np.c_[np.ones(len(d)), pd.get_dummies(d[['N_kg_ha','block_design']].astype(str), drop_first=True).to_numpy(float)]
    b = d.B_white.to_numpy(float)*100
    y = d.yield_kg_ha.to_numpy(float)
    rb=b-x@np.linalg.lstsq(x,b,rcond=None)[0]
    ry=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
    return x,rb,ry

def partial_statistics():
    pf=pd.read_csv(P/'plot_features.csv'); rows=[]; resid=[]; blocks=[]
    for dom,d in pf.groupby('domain'):
        x,rb,ry=partial_result(d); rank=np.linalg.matrix_rank(x)
        r=float(stats.pearsonr(rb,ry).statistic); df=len(d)-rank-1
        slope=float(rb@ry/(rb@rb)); s2=np.sum((ry-slope*rb)**2)/df
        rows.append(dict(domain=dom,n=len(d),control_rank=rank,df_resid=df,r=r,
                         P=2*stats.t.sf(abs(r*np.sqrt(df/(1-r*r))),df),slope=slope,
                         slope_se=np.sqrt(s2/(rb@rb))))
        resid.append(pd.DataFrame(dict(domain=dom,plot_id=d.plot_id,N=d.N_kg_ha,block=d.block_design,boll_residual_pct=rb,yield_residual=ry)))
        blocks.append(x)
    rr=pd.concat(resid,ignore_index=True);rb=rr.boll_residual_pct.to_numpy();ry=rr.yield_residual.to_numpy()
    rank=np.linalg.matrix_rank(block_diag(*blocks));df=len(rr)-rank-1;r=float(stats.pearsonr(rb,ry).statistic)
    slope=float(rb@ry/(rb@rb));s2=np.sum((ry-slope*rb)**2)/df
    rows.append(dict(domain='pooled',n=len(rr),control_rank=rank,df_resid=df,r=r,
                     P=2*stats.t.sf(abs(r*np.sqrt(df/(1-r*r))),df),slope=slope,slope_se=np.sqrt(s2/(rb@rb))))
    pd.DataFrame(rows).to_csv(OUT/'partial_correlations.csv',index=False)
    rr.to_csv(OUT/'partial_residuals.csv',index=False)
    return rows

def blocked_fit(d, response, pooled=False):
    d=d.copy();d['N_level']=d.N_kg_ha.astype(int).astype(str);d['block']=d.block_design.astype(str)
    nuisance='C(domain) + C(domain):C(block)' if pooled else 'C(block)'
    full=smf.ols(f'{response} ~ C(N_level) + {nuisance}',data=d).fit()
    reduced=smf.ols(f'{response} ~ {nuisance}',data=d).fit()
    f,p,dfn=full.compare_f_test(reduced)
    levels=sorted(d.N_kg_ha.unique()); vectors=[]; means=[]
    for n in levels:
        if pooled:
            grid=d[['domain','block']].drop_duplicates().copy()
        else:
            grid=pd.DataFrame({'block':sorted(d.block.unique())})
        grid['N_level']=str(int(n))
        v=np.asarray(build_design_matrices([full.model.data.design_info],grid)[0]).mean(axis=0)
        vectors.append(v);means.append(float(v@full.params))
    pairs=[]
    for i,j in itertools.combinations(range(len(levels)),2):
        c=vectors[i]-vectors[j];diff=float(c@full.params);se=float(np.sqrt(c@full.cov_params()@c))
        pairs.append(dict(i=i,j=j,N1=int(levels[i]),N2=int(levels[j]),difference=diff,se=se,
                          P_raw=float(2*stats.t.sf(abs(diff/se),full.df_resid))))
    adjusted=multipletests([r['P_raw'] for r in pairs],method='holm')[1]
    sig=np.zeros((len(levels),len(levels)),bool)
    for row,pa in zip(pairs,adjusted):
        row['P_Holm']=pa;sig[row['i'],row['j']]=sig[row['j'],row['i']]=pa<0.05
    # Insert-absorb compact letter display; only model contrasts determine sharing.
    columns=[set(range(len(levels)))]
    for i,j in itertools.combinations(range(len(levels)),2):
        if not sig[i,j]:continue
        new=[]
        for c in columns:
            new.extend([c-{i},c-{j}] if i in c and j in c else [c])
        columns=[]
        for c in new:
            if c and c not in columns:columns.append(c)
        columns=[c for c in columns if not any(c<other for other in columns)]
    columns.sort(key=lambda c:-max(means[i] for i in c))
    rows=[]
    for i,n in enumerate(levels):
        v=vectors[i];se=float(np.sqrt(v@full.cov_params()@v))
        raw=d.loc[d.N_kg_ha==n,response]
        rows.append(dict(N_str=str(int(n)),N_num=float(n),mean=means[i],se=se,
                         raw_mean=raw.mean(),raw_se=raw.sem(),n=len(raw),
                         cld=''.join(chr(97+k) for k,c in enumerate(columns) if i in c)))
    return pd.DataFrame(rows),float(f),float(p),full,pd.DataFrame(pairs)

def block_statistics():
    common=pd.read_csv(P/'common_plot_features_common_DAS.csv');tests=[];tables=[];pairs=[];curves=[]
    for dom,d in common.groupby('domain'):
        tab,f,p,fit,contr=blocked_fit(d,'yield_kg_ha');tab['domain']=dom;tables.append(tab)
        tests.append(dict(domain=dom,response='yield',F=f,P=p,df_num=6,df_den=fit.df_resid,n=len(d)))
        contr['domain']=dom;contr['response']='yield';pairs.append(contr)
        dd=d.copy();dd['N_scaled']=dd.N_kg_ha/450;dd['N2']=dd.N_scaled**2;dd['block']=dd.block_design.astype(str)
        quad=smf.ols('yield_kg_ha ~ N_scaled + N2 + C(block)',data=dd).fit()
        for n in np.linspace(0,450,100):
            grid=pd.DataFrame(dict(N_scaled=n/450,N2=(n/450)**2,block=sorted(dd.block.unique())))
            v=np.asarray(build_design_matrices([quad.model.data.design_info],grid)[0]).mean(axis=0)
            mean=float(v@quad.params);half=float(stats.t.ppf(.975,quad.df_resid)*np.sqrt(v@quad.cov_params()@v))
            curves.append(dict(domain=dom,N=n,mean=mean,lower=mean-half,upper=mean+half))
    pf=pd.read_csv(P/'plot_features.csv');pf['B_pct']=pf.B_white*100
    tab,f,p,fit,contr=blocked_fit(pf,'B_pct',pooled=True);tab['domain']='pooled';tab.to_csv(OUT/'boll_adjusted_means.csv',index=False)
    tests.append(dict(domain='pooled',response='boll',F=f,P=p,df_num=6,df_den=fit.df_resid,n=len(pf)))
    contr['domain']='pooled';contr['response']='boll';pairs.append(contr)
    # High-N slope is conditional on environment and its spatial blocks.
    high=pf[pf.N_kg_ha>=150].copy();high['N100']=high.N_kg_ha/100;high['block']=high.block_design.astype(str)
    hf=smf.ols('B_pct ~ N100 + C(domain) + C(domain):C(block)',data=high).fit()
    pd.DataFrame([dict(slope_percentage_points_per_100N=hf.params['N100'],P=hf.pvalues['N100'],
                      lower=hf.conf_int().loc['N100',0],upper=hf.conf_int().loc['N100',1],n=len(high))]).to_csv(OUT/'high_N_boll_slope.csv',index=False)
    pd.DataFrame(tests).to_csv(OUT/'blocked_omnibus_tests.csv',index=False)
    pd.concat(tables,ignore_index=True).to_csv(OUT/'yield_adjusted_means.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(OUT/'blocked_pairwise_contrasts.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'yield_quadratic_curves.csv',index=False)
    return tests

def extract_matched_indices():
    import geopandas as gpd
    import rasterio
    from extract_raw_rgb_information import crop, GPKG
    direct=pd.read_csv(P/'direct_image_features.csv');common=pd.read_csv(P/'common_plot_features_common_DAS.csv')
    keys=set(zip(common.domain,common.plot_id));use=[]
    for _,r in direct.iterrows():
        anchors=[92,107,130,142,181] if r.year==2023 else [92,106,128,141,181]
        if (f'{r.year}_{r.field}',r.plot_id) in keys and r.DAS in anchors:use.append(r)
    use=pd.DataFrame(use);geometry=gpd.read_file(GPKG,layer='inward_buffer_plot');rows=[]
    for k,(path,d) in enumerate(use.groupby('raster_path'),1):
        with rasterio.open(path) as src:
            gg=geometry.to_crs(src.crs)
            for row in d.itertuples():
                geom=gg[(gg.year==row.year)&(gg.field==row.field)&(gg.plot_id==row.plot_id)].geometry.iloc[0]
                rgb,valid=crop(src,geom);a=rgb[valid].astype(np.float64);a=a/a.sum(axis=1,keepdims=True);r,g,b=a.T
                def ratio(num,den):
                    ok=np.abs(den)>1e-9
                    return float(np.mean(num[ok]/den[ok]))
                vals=[np.mean(1.4*b-g),np.mean(2*g-r-b),np.mean(1.4*r-g),np.mean(3*g-2.4*r-b),
                      ratio(g,b),ratio(2*g-r-b,2*g+r+b),ratio(r-b,r+b),ratio(g*g-r*r,g*g+r*r),
                      ratio(g-b,g+b),ratio(g-r,g+r),ratio(g*g-r*b,g*g+r*b),ratio(r,g),ratio(g-r,g+r-b)]
                assert np.isfinite(vals).all();assert valid.sum()==row.valid_pixels
                rec=dict(domain=f'{row.year}_{row.field}',plot_id=row.plot_id,DAS=row.DAS,valid_pixels=int(valid.sum()),pixel_support='all_valid_inward_ROI',raster_path=path)
                rec.update(dict(zip(m.INDICES,vals)));rows.append(rec)
        print('MATCHED_INDEX_EXTRACTION',k,len(rows),flush=True)
    result=pd.DataFrame(rows);assert len(result)==64*5
    result.to_csv(OUT/'matched_direct_RGB_indices.csv',index=False,float_format='%.12g')
    return result

def matched_models(indices):
    common=pd.read_csv(P/'common_plot_features_common_DAS.csv');sets={};pred=[];tuning=[];support=[]
    for dom,d in common.groupby('domain'):
        d=d.reset_index(drop=True);idx=indices[indices.domain==dom].set_index(['plot_id','DAS']);x52=[];x13=[];x6=[]
        for row in d.itertuples():
            anchors=[92,107,130,142] if dom.startswith('2023') else [92,106,128,141]
            x52.append(np.concatenate([idx.loc[(row.plot_id,day),m.INDICES].to_numpy(float) for day in anchors]))
            x13.append(idx.loc[(row.plot_id,anchors[-1]),m.INDICES].to_numpy(float));x6.append(idx.loc[(row.plot_id,181),m.INDICES].to_numpy(float))
        n=m.design(d,'N');sets[dom]=(d,{'N_RGB52_common4':np.c_[n,x52],'N_RGB13_S5b':np.c_[n,x13],'N_RGB13_S6_matched':np.c_[n,x6]})
    for source,(d,designs) in sets.items():
        ii=np.arange(len(d));g=d.block_design.to_numpy();y=d.yield_kg_ha.to_numpy()
        jobs=[(ii[g!=b],source,ii[g==b],'BlockCV',int(b)) for b in np.unique(g)]
        jobs.extend((ii,t,np.arange(len(td)),'adjacent_field' if t[:4]==source[:4] else 'cross_year',0) for t,(td,_) in sets.items() if t!=source)
        for spec,x in designs.items():
            for train,target,test,ev,held in jobs:
                rank=[]
                for alpha in m.ALPHAS:
                    yh=np.full(len(train),np.nan)
                    for b in np.unique(g[train]):
                        tr=train[g[train]!=b];te=train[g[train]==b]
                        yh[g[train]==b]=m.PartialRidge(alpha).fit(x[tr],y[tr]).predict(x[te])
                    err=yh-y[train];rank.append((np.sqrt(np.mean(err**2)),np.mean(abs(err)),-alpha,alpha))
                _,_,_,alpha=min(rank);mod=m.PartialRidge(alpha).fit(x[train],y[train]);td,tx=sets[target]
                vals=mod.predict(tx[spec][test]);trainids=';'.join(map(str,d.iloc[train].plot_id.astype(int)))
                tuning.append(dict(source=source,target=target,specification=spec,evaluation=ev,held_block=held,train_n=len(train),train_plot_ids=trainids,alpha=alpha,inner_RMSE=min(rank)[0]))
                for j,yh in zip(test,vals):
                    r=td.iloc[j];pred.append(dict(variant='matched_direct_common64',source=source,target=target,algorithm='PartialRidge',specification=spec,evaluation=ev,plot_id=int(r.plot_id),block=int(r.block_design),N=float(r.N_kg_ha),observed=float(r.yield_kg_ha),predicted=float(yh),train_n=len(train),train_plot_ids=trainids,source_block_omitted=held))
    pp=pd.DataFrame(pred);met,within=m.summarize(pp)
    pp.to_csv(OUT/'matched_RGB_predictions.csv',index=False,float_format='%.12g')
    met.to_csv(OUT/'matched_RGB_metrics.csv',index=False,float_format='%.12g')
    within.to_csv(OUT/'matched_RGB_within_N.csv',index=False,float_format='%.12g')
    pd.DataFrame(tuning).to_csv(OUT/'matched_RGB_tuning.csv',index=False)
    return met

def main():
    raw=ROOT/'01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv';before=hashlib.sha256(raw.read_bytes()).hexdigest()
    partial=partial_statistics();blocked=block_statistics()
    if '--statistics-only' not in sys.argv:
        path=OUT/'matched_direct_RGB_indices.csv'
        indices=pd.read_csv(path) if path.exists() and '--extract' not in sys.argv else extract_matched_indices()
        matched=matched_models(indices)
        print(matched[['source','target','specification','RMSE']].to_string(index=False),flush=True)
    assert before==hashlib.sha256(raw.read_bytes()).hexdigest()
    receipt=dict(status='RECOMPUTED_CURRENT',raw_714_sha256=before,partial_correlations=partial,blocked_tests=blocked,
                 matched_support='64 identical plots; same training folds; all-valid approved ROI; common four dates; source-only scaling and inner tuning',
                 measurement_reference='AI extracted, author verified; not independent blind annotation')
    (OUT/'reanalysis_receipt.json').write_text(json.dumps(receipt,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    print('CURRENT_REANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
