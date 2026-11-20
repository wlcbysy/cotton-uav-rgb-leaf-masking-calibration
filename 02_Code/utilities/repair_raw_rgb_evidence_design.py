"""Review-directed comparisons: fixed estimand, matched support, source-only masks.

The main model is chosen for the management-conditional estimand, not target
scores. Human reference remains pending unless real completed labels exist.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/modeling'))
import mine_raw_rgb_information as m

OUT=m.OUT
SPECS=['N','N_Cearly','N_C','N_C_T']
CONDITIONS=['dense_DAS','common_DAS','dense_GDD','common_GDD']

def support_tables(direct,legacy,white='white_b140_s20'):
    base=m.build_table(direct,legacy,white);weather=pd.read_csv(m.ROOT/'01_Data/processed/weather/alar_2023_2024_daily_weather.csv').set_index('date')
    tables={};exclusions=[];validkeys=[]
    for _,r in base.iterrows():
        d=direct[(direct.year==r.year)&(direct.field==r.field)&(direct.plot_id==r.plot_id)]
        anchors=[92,107,130,142] if r.year==2023 else [92,106,128,141]
        missing=[s for s in anchors if s not in d.DAS.values]
        if missing:exclusions.append({'domain':r.domain,'plot_id':int(r.plot_id),'missing_direct_DAS':';'.join(map(str,missing))})
        else:validkeys.append((r.domain,int(r.plot_id)))
    base=base[[ (r.domain,int(r.plot_id)) in validkeys for r in base.itertuples()]].copy()
    for condition in CONDITIONS:
        tab=base.copy();rows=[]
        for ix,r in tab.iterrows():
            anchors=[92,107,130,142] if r.year==2023 else [92,106,128,141]
            d=direct[(direct.year==r.year)&(direct.field==r.field)&(direct.plot_id==r.plot_id)&direct.DAS.between(92,anchors[-1])].sort_values('DAS')
            if condition.startswith('common'):d=d[d.DAS.isin(anchors)]
            assert len(d)>=4
            if condition.endswith('GDD'):x=weather.loc[d.flight_date,'cum_GDD'].to_numpy(float)
            else:x=d.DAS.to_numpy(float)
            y=d.green_canopeo.to_numpy(float)
            assert np.all(np.diff(x)>0)
            early=np.flatnonzero(d.DAS.to_numpy()==anchors[-2])[0]
            tab.loc[ix,'T_auc_mean']=np.trapezoid(y,x)/(x[-1]-x[0])
            tab.loc[ix,'T_late_change']=(y[-1]-y[early])/(x[-1]-x[early])*(x[-1]-x[0])
            tab.loc[ix,'T_direct_dates']=len(d)
            rows.append({'condition':condition,'domain':r.domain,'plot_id':int(r.plot_id),'n_dates':len(d),'DAS':';'.join(map(str,d.DAS.astype(int))),'clock_values':';'.join(f'{v:.3f}' for v in x),'clock':'cum_GDD' if condition.endswith('GDD') else 'DAS'})
        tables[condition]=tab.reset_index(drop=True)
        pd.DataFrame(rows).to_csv(OUT/f'temporal_support_{condition}.csv',index=False)
    pd.DataFrame(exclusions).to_csv(OUT/'common_support_exclusions.csv',index=False)
    return tables

def add_with_trace(model,src,target,alg,spec,role,variant,train,source,omitted=0):
    start=len(m.PRED);m.add(model,src,target,alg,spec,role,variant)
    ids=';'.join(map(str,source.iloc[train].plot_id.astype(int)))
    for row in m.PRED[start:]:row.update(train_n=len(train),train_plot_ids=ids,source_block_omitted=int(omitted))

def run_support(tables):
    m.PRED=[];m.TUNE=[]
    for condition,table in tables.items():
        table.to_csv(OUT/f'common_plot_features_{condition}.csv',index=False,float_format='%.12g')
        sets={k:d.reset_index(drop=True) for k,d in table.groupby('domain')}
        for src,d in sets.items():
            ii=np.arange(len(d));groups=d.block_design.to_numpy()
            for alg in ['PartialRidge','Ridge','RF','OLS']:
                for spec in (['N'] if alg=='OLS' else SPECS):
                    for block in np.unique(groups):
                        train=ii[groups!=block];test=d.iloc[ii[groups==block]]
                        model=m.fit(d,alg,spec,train,condition,f'hold_block_{block}')
                        add_with_trace(model,src,test,alg,spec,'BlockCV',condition,train,d,block)
                        # Same exact frozen source training set, evaluated elsewhere.
                        for tar,target in sets.items():
                            if tar!=src:add_with_trace(model,src,target,alg,spec,f'matched_holdout_{block}',condition,train,d,block)
                    model=m.fit(d,alg,spec,ii,condition,'full_source')
                    for tar,target in sets.items():
                        if tar!=src:add_with_trace(model,src,target,alg,spec,'adjacent_field' if src[:4]==tar[:4] else 'cross_year',condition,ii,d)
            print('SUPPORT',condition,src,flush=True)
    pred=pd.DataFrame(m.PRED);met,within=m.summarize(pred)
    pred.to_csv(OUT/'support_predictions.csv',index=False,float_format='%.12g');met.to_csv(OUT/'support_metrics.csv',index=False,float_format='%.12g')
    within.to_csv(OUT/'support_within_N.csv',index=False,float_format='%.12g');pd.DataFrame(m.TUNE).to_csv(OUT/'support_tuning.csv',index=False,float_format='%.12g')
    # Source-fold metrics retain train composition, rather than pooling target re-use.
    keys=['variant','source','target','algorithm','specification','evaluation','source_block_omitted']
    rows=[]
    for key,d in pred[pred.source_block_omitted>0].groupby(keys,sort=False):
        err=d.predicted-d.observed;rows.append({**dict(zip(keys,key)),'train_n':int(d.train_n.iloc[0]),'train_plot_ids':d.train_plot_ids.iloc[0],'evaluation_n':len(d),'RMSE':np.sqrt(np.mean(err**2))})
    fold=pd.DataFrame(rows)
    for i,r in fold.iterrows():
        base=fold[(fold.variant==r.variant)&(fold.source==r.source)&(fold.target==r.target)&(fold.algorithm==r.algorithm)&(fold.evaluation==r.evaluation)&(fold.source_block_omitted==r.source_block_omitted)&(fold.specification=='N')]
        assert len(base)==1;fold.loc[i,'gain_RMSE']=base.iloc[0].RMSE-r.RMSE
    fold.to_csv(OUT/'matched_training_fold_metrics.csv',index=False,float_format='%.12g')
    return pred,met

def source_rule_choices():
    # These remain explicitly Codex reference points. Human validation is NOT inferred.
    scores=pd.read_csv(OUT/'mask_point_metrics.csv');points=pd.read_csv(OUT/'image_review/point_predictions.csv')
    choices=[]
    for (year,field),d in scores[scores.phase=='cotton'].groupby(['year','field']):
        chosen=d.sort_values(['balanced_accuracy','F1','rule'],ascending=[False,False,True]).iloc[0]
        pp=points[(points.year==year)&(points.field==field)&(points.phase=='cotton')&(points.rule==chosen.rule)]
        choices.append({'source':f'{year}_{field}','selected_rule':chosen.rule,'reference_type':'Codex_exploratory_source_image_points','calibration_domain':f'{year}_{field}','reference_points':len(pp),'reference_plot_ids':';'.join(map(str,sorted(pp.plot_id.unique()))),'reference_point_id_hash':hashlib.sha256(pp[['patch_id','point']].to_csv(index=False).encode()).hexdigest(),'target_image_labels_used_for_selection':False,'source_balanced_accuracy':chosen.balanced_accuracy})
    return pd.DataFrame(choices)

def run_source_only(direct,legacy):
    choices=source_rule_choices();choices.to_csv(OUT/'source_mask_calibration_roles.csv',index=False)
    m.PRED=[];m.TUNE=[]
    for _,r in choices.iterrows():
        # Common direct four-stage support is held constant for this comparison.
        tables=support_tables(direct,legacy,r.selected_rule);tab=tables['common_DAS']
        sets={k:d.reset_index(drop=True)for k,d in tab.groupby('domain')};src=r.source;source=sets[src];train=np.arange(len(source))
        assert set(map(int,r.reference_plot_ids.split(';')))<=set(source.plot_id.astype(int))
        for spec in ['N','N_B','N_C_B','N_C_T_B']:
            model=m.fit(source,'PartialRidge',spec,train,'source_image_calibration','full_source')
            for tar,target in sets.items():
                if tar==src:continue
                start=len(m.PRED);add_with_trace(model,src,target,'PartialRidge',spec,'adjacent_field' if src[:4]==tar[:4] else 'cross_year','source_image_calibration',train,source)
                for row in m.PRED[start:]:row.update(mask_calibration_domain=src,mask_rule=r.selected_rule,reference_type=r.reference_type,target_image_labels_used=False)
        print('SOURCE MASK',src,r.selected_rule,flush=True)
    pred=pd.DataFrame(m.PRED);met,within=m.summarize(pred)
    pred.to_csv(OUT/'source_role_predictions.csv',index=False,float_format='%.12g');met.to_csv(OUT/'source_role_metrics.csv',index=False,float_format='%.12g');within.to_csv(OUT/'source_role_within_N.csv',index=False,float_format='%.12g');pd.DataFrame(m.TUNE).to_csv(OUT/'source_role_tuning.csv',index=False,float_format='%.12g')
    return choices,pred,met

def main():
    direct=pd.read_csv(OUT/'direct_image_features.csv');legacy=pd.read_csv(m.ROOT/'01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv')
    policy={'status':'FIXED_FOR_REVIEW_DIRECTED_RETROSPECTIVE_COMPARISON','main_question':'Does image information beyond N predict within-management plot differences, rather than only field/treatment-level calibration?','main_model':'PartialRidge with unpenalized intercept,N,N²; optical terms source-scaled and penalized. Chosen for estimand after review, not prospective preregistration.','main_reference':'OLS quadratic N on exactly the same rows, source splits and targets','primary_specs':SPECS,'secondary_models':['Ridge','fixed_complexity_RF'],'temporal_design':'2x2: dense/common4directstages × DAS/GDD, identical complete-case plot support','pixel_reference_status':'Human reference pending; fixed Canopeo proxy has no learned image labels. Existing globally F1-selected cotton results remain exploration only. Source-specific cotton comparison uses source-only Codex labels and remains provisional.','cotton_BlockCV_status':'Not promoted to source-only validation: existing Codex labels are first-block-only; independent human calibration across blocks required.','human_reference_UI':'human_reference/index.html; user volunteered to annotate personally','raw_inputs_modified':False,'original_submission_package_modified':False}
    (OUT/'review_repair_protocol.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
    tables=support_tables(direct,legacy);p1,m1=run_support(tables);choices,p2,m2=run_source_only(direct,legacy)
    # Restore main comparison tables: source-mask candidates must not overwrite them.
    tables=support_tables(direct,legacy)
    receipt={'status':'COMPUTATIONAL_ANALYSES_COMPLETE / HUMAN_REFERENCE_PENDING','main_plot_records':len(tables['common_DAS']),'support_prediction_rows':len(p1),'support_metric_sets':len(m1),'source_role_prediction_rows':len(p2),'source_role_metric_sets':len(m2),'main_source_only_preprocessing':'fixed published green inequalities; no target image reference labels used','cotton_source_only_preprocessing':'source-specific Codex reference points; no target image labels or target yields used','human_reference_completed':False,'BlockCV_cotton_validation_complete':False,'publication_readiness':'not established by numerical QA'}
    (OUT/'review_repair_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
