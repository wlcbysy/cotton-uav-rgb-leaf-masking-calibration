"""Check accepted masks and reproduce confusion counts without changing labels."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

R=Path(__file__).resolve().parents[2]
F=R/'10_Results/cotton_rgb'/(R/'10_Results/cotton_rgb/CURRENT_FORMAL_VERSION.txt').read_text().strip()
A=R/'00_Project/annotation'
P=F/'analysis/data_mining_current'

def verify():
    declaration=json.loads((A/'annotation_declaration.json').read_text())
    if not declaration['human_completed']:
        assert not (P/'human_boll_pixel_metrics.csv').exists()
        return {'status':'PENDING','plots':21}
    assert declaration['human_reviewed'] and declaration['ai_assisted']
    assert declaration['independent_unassisted_annotation'] is False
    assert declaration['label_provenance'] == 'AI-assisted labels accepted unchanged after user-reported review'
    state=json.loads((A/'ai_review/review_state.json').read_text())['items']
    manifest=pd.read_csv(A/'sampling_manifest.csv')
    table=pd.read_csv(P/'human_boll_pixel_metrics.csv')
    assert len(manifest)==len(state)==21 and len(table)==21*12
    assert not table.duplicated(['image_id','rule']).any()
    assert table.ai_assisted.all()
    for r in manifest.itertuples():
        rgbpath=A/'images'/f'{r.image_id}.png'
        assert hashlib.sha256(rgbpath.read_bytes()).hexdigest()==r.rgb_sha256
        path=A/'completed_masks'/f'{r.image_id}.png';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert state[r.image_id]['reviewed'] and state[r.image_id]['mask_sha256']==digest
        # User accepted these exact drafts unchanged; retain a machine-label audit trail.
        assert path.read_bytes()==(A/'ai_draft_masks'/path.name).read_bytes()
        lab=np.asarray(Image.open(path));valid=np.asarray(Image.open(A/'valid_masks'/path.name))>0
        assert lab.shape==(384,384) and lab.dtype==np.uint8 and set(np.unique(lab))<={0,1,255}
        assert np.all(lab[~valid]==255)
        rgb=np.asarray(Image.open(rgbpath).convert('RGB')).astype(float)
        brightness=rgb.mean(axis=2);mx=rgb.max(axis=2);sat=(mx-rgb.min(axis=2))/np.maximum(mx,1)
        known=valid&(lab!=255);truth=lab==1
        for z in table.loc[table.image_id.eq(r.image_id)].itertuples():
            _,b,s=z.rule.split('_');pred=(brightness>=int(b[1:]))&(sat<=int(s[1:])/100)&valid
            counts=[int((known&truth&pred).sum()),int((known&~truth&pred).sum()),int((known&truth&~pred).sum()),int((known&~truth&~pred).sum())]
            assert counts==[z.TP,z.FP,z.FN,z.TN]
            tp,fp,fn,tn=counts
            ratio=lambda a,b:a/b if b else np.nan
            np.testing.assert_allclose([z.precision,z.recall,z.F1,z.IoU],[ratio(tp,tp+fp),ratio(tp,tp+fn),ratio(2*tp,2*tp+fp+fn),ratio(tp,tp+fp+fn)],equal_nan=True)
            assert z.labelled_pixels==int(known.sum()) and z.valid_pixels==int(valid.sum())
            np.testing.assert_allclose(z.unknown_fraction,1-known.sum()/valid.sum())
            assert z.mask_sha256==digest
    summary=pd.read_csv(P/'reviewed_boll_pixel_summary.csv')
    for row in summary.itertuples():
        d=table.loc[table.rule.eq(row.rule)]
        if row.domain!='All 21 plots':d=d.loc[d.domain.eq(row.domain)]
        assert row.plots==len(d)
        for key in ['precision','recall','F1','IoU','unknown_fraction']:
            np.testing.assert_allclose(getattr(row,key),d[key].mean())
    audit=json.loads((P/'reviewed_boll_pixel_audit.json').read_text())
    d=table.query("rule=='white_b140_s30'")
    for key in ['precision','recall','F1','IoU']:
        np.testing.assert_allclose(audit['macro_metrics'][key],d[key].mean())
    assert audit['counts']=={k:int(d[k].sum()) for k in ['TP','FP','FN','TN','labelled_pixels','valid_pixels','unknown_pixels']}
    return {'status':'PASS','plots':21,'rule_plot_scores':252,'reference':'AI-assisted; user chat-confirmed unchanged','independent_accuracy_claimed':False,'no_model_fits':True}

if __name__=='__main__':
    result=verify();(F/'qa/BOLL_PIXEL_REVIEW_QA_CURRENT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
