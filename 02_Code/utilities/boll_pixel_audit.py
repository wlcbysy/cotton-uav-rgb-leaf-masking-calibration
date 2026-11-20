"""Prepare crops; score explicitly accepted labels while retaining their provenance."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
F=ROOT/'10_Results/cotton_rgb'/(ROOT/'10_Results/cotton_rgb/CURRENT_FORMAL_VERSION.txt').read_text().strip()
OUT=F/'analysis/data_mining_current';DEST=ROOT/'00_Project/annotation'
sys.path.insert(0,str(ROOT/'02_Code/src/data'))
from extract_raw_rgb_information import crop,rules,GPKG

def prepare():
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds
    DEST.mkdir(exist_ok=True)
    for sub in ['images','valid_masks','mask_templates','completed_masks']:(DEST/sub).mkdir(exist_ok=True)
    direct=pd.read_csv(F/'analysis/raw_rgb_mining/direct_image_features.csv').query('DAS==181')
    geometry=gpd.read_file(GPKG,layer='inward_buffer_plot')
    manifest=[];sensitivity=[]
    for (year,field),d in direct.groupby(['year','field']):
        for j,(n,z) in enumerate(d.groupby('N_kg_ha')):
            desired=1+j%3
            z=z.assign(distance=abs(z.block_design-desired)).sort_values(['distance','plot_id']);r=z.iloc[0]
            with rasterio.open(r.raster_path) as src:
                g=geometry.loc[geometry.year.eq(year)&geometry.field.eq(field)&geometry.plot_id.eq(r.plot_id)].to_crs(src.crs).geometry.iloc[0]
                rgb,valid=crop(src,g);window=from_bounds(*g.bounds,src.transform).round_offsets().round_lengths()
                cy,cx=np.array(valid.shape)//2;h=w=384;y0=cy-h//2;x0=cx-w//2
                tile=rgb[y0:y0+h,x0:x0+w].astype('uint8');mask=valid[y0:y0+h,x0:x0+w]
                assert tile.shape==(h,w,3) and mask.any()
                image_id=f'{year}_{field}_N{int(n):03d}_P{int(r.plot_id):02d}'
                Image.fromarray(tile).save(DEST/'images'/f'{image_id}.png')
                Image.fromarray((mask*255).astype('uint8')).save(DEST/'valid_masks'/f'{image_id}.png')
                # 255 is explicitly unlabelled/unknown, never a negative class.
                Image.fromarray(np.full((h,w),255,dtype='uint8')).save(DEST/'mask_templates'/f'{image_id}.png')
                manifest.append(dict(image_id=image_id,domain=f'{year}_{field}',year=year,field=field,N=n,plot_id=int(r.plot_id),block=int(r.block_design),date=r.flight_date,DAS=181,raster_path=r.raster_path,source_row=int(window.row_off)+y0,source_col=int(window.col_off)+x0,width=w,height=h,valid_pixels=int(mask.sum()),rgb_sha256=hashlib.sha256((DEST/'images'/f'{image_id}.png').read_bytes()).hexdigest(),status='AWAITING_HUMAN'))
                baseline=rules(tile,mask)
                for brightness in [-20,0,20]:
                    for contrast in [.8,1.,1.2]:
                        perturbed=np.clip((tile.astype(float)-127.5)*contrast+127.5+brightness,0,255).astype('uint8')
                        for rule,pred in rules(perturbed,mask).items():
                            if not rule.startswith('white_'):continue
                            fraction=float(pred[mask].mean());old=float(baseline[rule][mask].mean())
                            sensitivity.append(dict(image_id=image_id,domain=f'{year}_{field}',N=n,rule=rule,brightness_DN=brightness,contrast=contrast,original_fraction=old,perturbed_fraction=fraction,delta_percentage_points=100*(fraction-old),pixel_flip_fraction=float((pred[mask]!=baseline[rule][mask]).mean())))
    pd.DataFrame(manifest).to_csv(DEST/'sampling_manifest.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'boll_brightness_contrast_sensitivity.csv',index=False)
    declaration=DEST/'annotation_declaration.json'
    if not declaration.exists():declaration.write_text(json.dumps({'human_completed':False,'annotator':'','notes':'Set human_completed true only after completing the masks yourself.'},indent=2)+'\n')
    (DEST/'README.md').write_text('''# 开铃像素人工标注

本批 21 个小区覆盖三块田 × 七个 N 水平，均为 10 月 13 日（181 DAS）晚季原始影像。按 N 水平轮换优先区组，再按小区号固定选择；不按产量或阈值效果选图。每张为 inward-buffer ROI 中心的 384 × 384 原生像素，未拉伸颜色。

1. 在 `images/` 查看 RGB，在图像标注软件中打开对应的 `mask_templates/` 灰度 PNG。逐像素或多边形标记可见的白色棉纤维；亮土、膜、反光和叶片属于背景。不能可靠区分时保留未知。
2. 灰度值约定：**1 = 明确可见开铃棉纤维，0 = 明确背景，255 = 未标注/不确定**。模板全部为 255，仅为空白画布，不是人工参考标签。`valid_masks/` 中 0 的无效区域必须保持未知；边界混合像素可保留 255。
3. 将完成的单通道 PNG 放入 `completed_masks/`，文件名与原图相同；保持 384 × 384，不缩放、不插值。不要将模板原样标为已完成。建议先完整标注少量切片检查操作，再完成全部。
4. 完成后在 `annotation_declaration.json` 填写真实标注者，将 `human_completed` 改为 true。运行 `python3 02_Code/utilities/boll_pixel_audit.py score`。脚本检查尺寸、值域、有效区域和未知比例，保留每张图的混淆计数，计算 precision、recall、F1、IoU；分母为零时记为缺失值。

指标只代表这些中心切片，不能当作全小区开铃面积精度。未知像素从指标分母排除并报告比例；不同小区不能被像素数量伪装成大量独立重复。当前未生成任何真实标注，也没有计算人工验证精度。
''')
    print('Prepared',len(manifest),'unlabelled plots; no human labels generated')

def score():
    declaration=json.loads((DEST/'annotation_declaration.json').read_text())
    if declaration.get('human_completed') is not True or not declaration.get('annotator','').strip():
        print('AWAITING_HUMAN: no metrics calculated');return
    manifest=pd.read_csv(DEST/'sampling_manifest.csv');rows=[]
    for r in manifest.itertuples():
        path=DEST/'completed_masks'/f'{r.image_id}.png'
        if not path.exists():raise ValueError(f'Missing human mask: {path}')
        rgbpath=DEST/'images'/f'{r.image_id}.png'
        assert hashlib.sha256(rgbpath.read_bytes()).hexdigest()==r.rgb_sha256
        lab=np.asarray(Image.open(path));valid=np.asarray(Image.open(DEST/'valid_masks'/f'{r.image_id}.png'))>0
        assert lab.ndim==2 and lab.shape==valid.shape and set(np.unique(lab))<={0,1,255}
        assert np.all(lab[~valid]==255),'Outside valid ROI must remain unknown'
        keep=valid&(lab!=255);assert keep.any(),'Empty unlabelled template is not a completed mask'
        rgb=np.asarray(Image.open(rgbpath).convert('RGB'))
        for rule,pred in rules(rgb,valid).items():
            if not rule.startswith('white_'):continue
            y=lab[keep]==1;p=pred[keep];tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
            ratio=lambda a,b:a/b if b else np.nan
            rows.append(dict(image_id=r.image_id,domain=r.domain,N=r.N,rule=rule,TP=tp,FP=fp,FN=fn,TN=tn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),F1=ratio(2*tp,2*tp+fp+fn),IoU=ratio(tp,tp+fp+fn),labelled_pixels=int(keep.sum()),valid_pixels=int(valid.sum()),unknown_pixels=int((valid&~keep).sum()),unknown_fraction=1-keep.sum()/valid.sum(),annotator=declaration['annotator'],mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),label_provenance=declaration.get('label_provenance','human annotation'),ai_assisted=declaration.get('ai_assisted',False),confirmation_channel=declaration.get('confirmation_channel','recorded annotation')))
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/'human_boll_pixel_metrics.csv',index=False)
    # Equal weight per physical plot, never treat pixels as independent replicates.
    summaries=[]
    for rule,d in metrics.groupby('rule'):
        for domain,z in [('All 21 plots',d),*list(d.groupby('domain'))]:
            summaries.append(dict(rule=rule,domain=domain,plots=len(z),
                **{k:float(z[k].mean()) for k in ['precision','recall','F1','IoU','unknown_fraction']},
                unknown_min=float(z.unknown_fraction.min()),unknown_max=float(z.unknown_fraction.max()),
                **{k:int(z[k].sum()) for k in ['TP','FP','FN','TN','labelled_pixels','valid_pixels','unknown_pixels']}))
    pd.DataFrame(summaries).to_csv(OUT/'reviewed_boll_pixel_summary.csv',index=False)
    selected=metrics.query("rule=='white_b140_s30'")
    # Audit the excluded set explicitly: a perfect conditional score does not
    # establish performance on the pixels that the draft left unknown.
    withheld_positive=0
    for r in manifest.itertuples():
        lab=np.asarray(Image.open(DEST/'completed_masks'/f'{r.image_id}.png'))
        valid=np.asarray(Image.open(DEST/'valid_masks'/f'{r.image_id}.png'))>0
        rgb=np.asarray(Image.open(DEST/'images'/f'{r.image_id}.png').convert('RGB'))
        withheld_positive+=int((rules(rgb,valid)['white_b140_s30'] & valid & (lab==255)).sum())
    report=dict(status='SCORED_REVIEWED_LABELS',primary_rule='white_b140_s30',primary_rule_basis='Unchanged rule illustrated in Fig. 5a; not selected on the new scores.',
        plots=len(selected),rules=metrics.rule.nunique(),label_provenance=declaration.get('label_provenance'),
        ai_assisted=declaration.get('ai_assisted',False),independent_unassisted_annotation=declaration.get('independent_unassisted_annotation',False),
        confirmation_channel=declaration.get('confirmation_channel'),confirmation_quote=declaration.get('confirmation_quote'),
        macro_metrics={k:float(selected[k].mean()) for k in ['precision','recall','F1','IoU']},
        mean_unknown_fraction=float(selected.unknown_fraction.mean()),unknown_min=float(selected.unknown_fraction.min()),unknown_max=float(selected.unknown_fraction.max()),
        counts={k:int(selected[k].sum()) for k in ['TP','FP','FN','TN','labelled_pixels','valid_pixels','unknown_pixels']},
        rule_positive_pixels_withheld_as_unknown=withheld_positive,
        interpretation='Conditional agreement with AI-assisted labels accepted unchanged by the user. Shared colour cues and selective withholding of ambiguous pixels preclude independent accuracy claims. Unknown pixels are excluded, not classified as background.',
        no_model_fitting=True)
    (OUT/'reviewed_boll_pixel_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Reviewed-label scores saved with explicit provenance and unknown-pixel audit; not independent accuracy.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','score']);args=ap.parse_args();globals()[args.action]()
