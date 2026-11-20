"""Consume actual human labels and run fully source-calibrated cotton comparisons.

Missing labels produce a pending status, never fabricated classes or zero scores.
Calibration/validation roles are assigned to whole physical plots beforehand.
Mask selection is repeated inside inner and outer SOURCE training splits.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import repair_raw_rgb_evidence_design as repair
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/data'))
from extract_raw_rgb_information import rules
m=repair.m;OUT=m.OUT;REF=OUT/'human_reference'
PREFIX='human'

def load_reference(file=None,independent_human=True):
    data=json.loads((REF/'review_dataset.json').read_text());ids={p['id']for r in data['records']for p in r['points']};file=Path(file)if file else REF/'human_labels.json'
    status={'status':'AWAITING_HUMAN_REFERENCE','dataset_id':data['dataset_id'],'required_points':len(ids),'labelled_points':0,'human_reference_completed':False}
    if not file.exists():return None,status
    doc=json.loads(file.read_text())
    assert doc.get('dataset_id')==data['dataset_id'],'Human reference dataset differs from prepared images'
    if independent_human:
        assert doc.get('human_declared') is True and doc.get('annotator','').strip(),'Actual human declaration and name required'
    else:
        assert doc.get('human_declared') is False and doc.get('independent_human_reference') is False and doc.get('annotator')=='Codex','AI provenance must remain explicit'
    labels=doc.get('labels',{});assert set(labels)<=ids and all(v in ['g','c','b','u']for v in labels.values())
    status.update(labelled_points=len(labels),annotator=doc['annotator'],uncertain_points=sum(v=='u' for v in labels.values()))
    if set(labels)!=ids:status['status']='PARTIAL_HUMAN_REFERENCE';return None,status
    if independent_human and (REF/'display_verification.json').exists():
        records={r['id']:r for r in data['records']};checks=doc.get('display_checks',{});verified=set()
        for pid,c in checks.items():
            r=records.get(pid.split('_')[0])
            if r and c.get('image_id')==r['id']and c.get('width')==r['width']and c.get('height')==r['height']:verified.add(pid)
        status.update(human_input_complete=True,display_verified_points=len(verified))
        if verified!=ids:
            status.update(status='HUMAN_LABELS_SAVED / DISPLAY_ALIGNMENT_RECHECK_REQUIRED',analysis_status='Known UI alignment defect: raw labels preserved; independent validation requires verified display correspondence.')
            return None,status
    private=pd.read_csv(REF/'private_sampling_manifest.csv').set_index('id');rows=[]
    for r in data['records']:
        rgb=np.array(Image.open(REF/r['image']).convert('RGB'));ms=rules(rgb,np.ones(rgb.shape[:2],dtype=bool));meta=private.loc[r['id']].to_dict();total=sum(p['stratum_valid_pixels']for p in r['points'])
        assert total==int(meta['valid_pixels'])
        for p in r['points']:
            row={**meta,'image_id':r['id'],'point_id':p['id'],'x':p['x'],'y':p['y'],'label':labels[p['id']],'area_weight':p['stratum_valid_pixels']/total,'independent_human_reference':independent_human}
            row.update({name:bool(mask[p['y'],p['x']])for name,mask in ms.items()});rows.append(row)
    status.update(status='HUMAN_REFERENCE_COMPLETE_INPUT_VALIDATED'if independent_human else'AI_REFERENCE_COMPLETE_INPUT_VALIDATED',human_reference_completed=independent_human,AI_reference_completed=not independent_human,label_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    points=pd.DataFrame(rows);points.to_csv(OUT/f'{PREFIX}_reference_points.csv',index=False)
    return points,status

def metrics(y,p,w):
    y=np.asarray(y,bool);p=np.asarray(p,bool);w=np.asarray(w,float)
    tp=w[y&p].sum();tn=w[~y&~p].sum();fp=w[~y&p].sum();fn=w[y&~p].sum()
    recall=tp/(tp+fn)if tp+fn else np.nan;specificity=tn/(tn+fp)if tn+fp else np.nan
    return {'precision':tp/(tp+fp)if tp+fp else 0.,'recall':recall,'specificity':specificity,'balanced_accuracy':(recall+specificity)/2,'F1':2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else 0.,'point_n':len(y)}

def select_rule(points,source,train_plots):
    d=points[(points.domain==source)&(points.phase=='cotton')&(points.role=='calibration')&points.plot_id.isin(train_plots)&(points.label!='u')]
    if d.empty or d.label.eq('c').nunique()!=2:raise ValueError(f'No two-class source calibration support for {source},plots={list(train_plots)}')
    rank=[]
    for rule in [c for c in d if c.startswith('white_')]:
        score=metrics(d.label.eq('c'),d[rule],d.area_weight)
        rank.append((-score['balanced_accuracy'],-score['F1'],rule))
    rule=min(rank)[2]
    return rule,{'calibration_domain':source,'calibration_plot_ids':';'.join(map(str,sorted(d.plot_id.unique()))),'calibration_image_ids':';'.join(sorted(d.image_id.unique())),'calibration_point_n':len(d)}

def image_validation(points,direct,independent_human=True):
    rows=[];areas=[];sources=points.domain.unique()
    for source in sources:
        rule,trace=select_rule(points,source,points[points.domain==source].plot_id.unique())
        for domain in points.domain.unique():
            for phase,mask,positive in [('cotton',rule,'c'),('canopy','green_canopeo','g')]:
                z=points[(points.domain==domain)&(points.phase==phase)&(points.role=='validation')]
                known=z[z.label!='u'];score=metrics(known.label.eq(positive),known[mask],known.area_weight)
                rows.append({'calibration_source':source,'validation_domain':domain,'phase':phase,'mask_rule':mask,**score,'independent_human_reference':independent_human,**trace})
                for image,d in z.groupby('image_id'):
                    w=d.area_weight;lower=float(w[d.label==positive].sum());unknown=float(w[d.label=='u'].sum());r=d.iloc[0]
                    target=direct[(direct.year==r.year)&(direct.field==r.field)&(direct.plot_id==r.plot_id)&(direct.DAS==r.DAS)].iloc[0];pred=float(target[mask])
                    areas.append({'calibration_source':source,'validation_domain':domain,'image_id':image,'phase':phase,'mask_rule':mask,'plot_id':int(r.plot_id),'reference_fraction_lower':lower,'reference_fraction_upper':lower+unknown,'reference_uncertain_weight':unknown,'predicted_full_ROI_fraction':pred,'absolute_error_to_reference_interval':max(lower-pred,pred-lower-unknown,0),'reference_type':'full_ROI_stratified_point_area_estimate_not_dense_polygon_truth'})
    pd.DataFrame(rows).to_csv(OUT/f'{PREFIX}_mask_validation_metrics.csv',index=False,float_format='%.12g');pd.DataFrame(areas).to_csv(OUT/f'{PREFIX}_ROI_area_validation.csv',index=False,float_format='%.12g')

def nested_cotton_models(points,direct,legacy):
    base=repair.support_tables(direct,legacy)['common_DAS'];sets={k:d.reset_index(drop=True)for k,d in base.groupby('domain')}
    # Cache ONLY the changing white column; common green/time support stays fixed.
    b=direct[direct.DAS==181].set_index(['year','field','plot_id']);cache={};rule_cache={};tuning=[];preds=[];failures=[]
    def table(rule,domain):
        key=(rule,domain)
        if key not in cache:
            d=sets[domain].copy();d['B_white']=[b.loc[(int(r.year),r.field,int(r.plot_id)),rule]for r in d.itertuples()];cache[key]=d
        return cache[key]
    def choose(source,idx):
        plots=tuple(sorted(sets[source].iloc[idx].plot_id.astype(int)));key=(source,plots)
        if key not in rule_cache:rule_cache[key]=select_rule(points,source,plots)
        return rule_cache[key]
    def fit(source,train,spec,role):
        d=sets[source];y=d.yield_kg_ha.to_numpy(float);groups=d.block_design.to_numpy()
        if spec=='N':return m.PartialRidge(100000.).fit(m.design(d.iloc[train],spec),y[train]),'not_used',{}
        rank=[];inner_trace=[]
        for alpha in m.ALPHAS:
            yh=np.full(len(train),np.nan)
            for block in np.unique(groups[train]):
                tr=train[groups[train]!=block];te=train[groups[train]==block];rule,trace=choose(source,tr)
                if alpha==m.ALPHAS[0]:
                    inner_trace.append({'held_source_block':int(block),'train_plot_ids':';'.join(map(str,d.iloc[tr].plot_id.astype(int))),'test_plot_ids':';'.join(map(str,d.iloc[te].plot_id.astype(int))),'mask_rule':rule,**trace})
                mod=m.PartialRidge(alpha).fit(m.design(table(rule,source).iloc[tr],spec),y[tr]);yh[groups[train]==block]=mod.predict(m.design(table(rule,source).iloc[te],spec))
                assert set(map(int,trace['calibration_plot_ids'].split(';')))<=set(d.iloc[tr].plot_id.astype(int))
            e=yh-y[train];rank.append((float(np.sqrt(np.mean(e**2))),float(np.mean(abs(e))),-alpha,alpha))
        score,mae,_,alpha=min(rank);rule,trace=choose(source,train)
        mod=m.PartialRidge(alpha).fit(m.design(table(rule,source).iloc[train],spec),y[train])
        tuning.append({'source':source,'role':role,'specification':spec,'alpha':alpha,'inner_RMSE':score,'mask_rule':rule,'inner_mask_roles':json.dumps(inner_trace,sort_keys=True),**trace})
        return mod,rule,trace
    for source,d in sets.items():
        ii=np.arange(len(d));groups=d.block_design.to_numpy()
        for spec in ['N','N_B','N_C_B','N_C_T_B']:
            jobs=[(ii[groups!=block],source,ii[groups==block],'BlockCV',int(block))for block in np.unique(groups)]
            jobs +=[(ii,target,np.arange(len(t)),'adjacent_field' if source[:4]==target[:4]else 'cross_year',0)for target,t in sets.items()if target!=source]
            for train,target,test,role,block in jobs:
                try:mod,rule,trace=fit(source,train,spec,f'{role}_hold_{block}')
                except ValueError as e:failures.append({'source':source,'target':target,'specification':spec,'evaluation':role,'omitted_block':block,'reason':str(e)});continue
                td=sets[target]if spec=='N'else table(rule,target);scores=mod.predict(m.design(td.iloc[test],spec))
                assert set(d.iloc[train].plot_id.astype(int)).isdisjoint(set(td.iloc[test].plot_id.astype(int)))if source==target else True
                for j,yh in zip(test,scores):
                    r=td.iloc[j];preds.append({'variant':f'{PREFIX}_source_nested_common_DAS','source':source,'target':target,'algorithm':'PartialRidge','specification':spec,'evaluation':role,'plot_id':int(r.plot_id),'block':int(r.block_design),'N':float(r.N_kg_ha),'observed':float(r.yield_kg_ha),'predicted':float(yh),'train_n':len(train),'train_plot_ids':';'.join(map(str,d.iloc[train].plot_id.astype(int))),'mask_rule':rule,'source_block_omitted':block,'target_reference_labels_used':False,**trace})
    p=pd.DataFrame(preds)
    # Never present partial outer-fold predictions as complete BlockCV scores.
    keep=[]
    for key,d in p.groupby(['source','target','specification','evaluation']):
        if key[3]=='BlockCV'and len(d)!=len(sets[key[0]]):continue
        keep.extend(d.index.tolist())
    scored=p.loc[keep];met,within=m.summarize(scored)
    p.to_csv(OUT/f'{PREFIX}_source_predictions.csv',index=False,float_format='%.12g');met.to_csv(OUT/f'{PREFIX}_source_model_metrics.csv',index=False,float_format='%.12g');within.to_csv(OUT/f'{PREFIX}_source_within_N.csv',index=False,float_format='%.12g');pd.DataFrame(tuning).to_csv(OUT/f'{PREFIX}_source_tuning.csv',index=False,float_format='%.12g')
    (OUT/f'{PREFIX}_source_unavailable.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
    return {'prediction_rows':len(p),'complete_metric_sets':len(met),'unavailable_jobs':len(failures)}

def main():
    points,status=load_reference()
    (REF/'reference_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    if points is None:print(json.dumps(status,ensure_ascii=False));return
    direct=pd.read_csv(OUT/'direct_image_features.csv');legacy=pd.read_csv(m.ROOT/'01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv')
    image_validation(points,direct);result=nested_cotton_models(points,direct,legacy)
    status.update(status='HUMAN_INPUT_ANALYZED_MEASUREMENT_ACCURACY_REPORTED_NOT_ASSUMED',**result)
    (REF/'reference_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
