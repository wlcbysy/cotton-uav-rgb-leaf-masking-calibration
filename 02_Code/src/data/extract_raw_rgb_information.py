"""Extract yield-blind RGB coverage information from the current 33 orthomosaics.

Raw inputs and the locked 714-slot table are never overwritten. All coordinates
are reprojected to each raster CRS. Denominators are valid pixels INSIDE the
approved 0.5-m inward ROI, not its rectangular bounding box.
"""
from pathlib import Path
import argparse, hashlib, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, transform as win_transform
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT/'10_Results/cotton_rgb'/(ROOT/'10_Results/cotton_rgb/CURRENT_FORMAL_VERSION.txt').read_text().strip()
OUT = FORMAL/'analysis/raw_rgb_mining'
QA = OUT/'image_review'
INPUT = ROOT/'01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv'
GPKG = ROOT/'01_Data/processed/spatial/cotton_rgb/field_authority.gpkg'
warnings.filterwarnings('ignore',category=rasterio.errors.NodataShadowWarning)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def crop(src,geom):
    w=from_bounds(*geom.bounds,src.transform).round_offsets().round_lengths()
    a=src.read([1,2,3],window=w)
    inside=geometry_mask([geom],out_shape=a.shape[1:],transform=win_transform(w,src.transform),invert=True)
    valid=inside & (a.astype(np.int32).sum(axis=0)>0)
    if src.nodata is not None: valid &= np.all(a!=src.nodata,axis=0)
    if src.count>=4: valid &= src.read(4,window=w)>0
    return np.moveaxis(a,0,2),valid

def arrays(rgb):
    a=rgb.astype(np.float32); r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
    mx=a.max(axis=2);mn=a.min(axis=2)
    sat=np.divide(mx-mn,mx,out=np.zeros_like(mx),where=mx>0)
    brightness=a.mean(axis=2)
    return r,g,b,sat,brightness

def rules(rgb,valid):
    r,g,b,sat,br=arrays(rgb)
    # Original Canopeo color inequalities (no connected-component modification).
    green=(r<0.95*g)&(b<0.95*g)&(2*g-r-b>20)&valid
    res={'green_canopeo':green,
         'green_loose':(r<g)&(b<g)&(2*g-r-b>10)&valid,
         'green_strict':(r<0.90*g)&(b<0.90*g)&(2*g-r-b>30)&valid}
    for t in [120,140,160,180]:
        for s in [0.20,0.30,0.40]:
            res[f'white_b{t}_s{int(s*100)}']=(br>=t)&(sat<=s)&valid
    return res

def prepare_review(df,gdf):
    """Fixed geometry-only crop selection; no yield or segmentation labels shown."""
    QA.mkdir(parents=True,exist_ok=True)
    meta=[]
    # Three N levels; one plot per level in first block, fixed before outcomes.
    for yr,field in [(2024,'F1'),(2024,'F2'),(2023,'F1')]:
        for phase,das in [('cotton',181),('canopy',128 if yr==2024 else 130)]:
            sub=df[(df.year==yr)&(df.field==field)&(df.DAS==das)]
            selected=[]
            for n in [0,225,450]:
                row=sub[sub.N_kg_ha==n].sort_values(['block_design','plot_id']).iloc[0]
                selected.append(row)
            tiles=[]
            for k,row in enumerate(selected):
                geom=gdf[(gdf.year==yr)&(gdf.field==field)&(gdf.plot_id==int(row.plot_id))]
                with rasterio.open(row.raster_path) as src:
                    rgb,valid=crop(src,geom.to_crs(src.crs).geometry.iloc[0])
                # Central 384 px native window. Regular 6x6 point lattice.
                cy,cx=np.array(valid.shape)//2
                rgb=rgb[cy-192:cy+192,cx-192:cx+192];valid=valid[cy-192:cy+192,cx-192:cx+192]
                name=f'{yr}_{field}_{phase}_p{int(row.plot_id):02d}'
                rgb=rgb.copy();rgb[~valid]=0
                assert rgb.max()<=255, 'Input DN scale must be checked before display or thresholds'
                Image.fromarray(rgb.astype(np.uint8)).save(QA/f'{name}.png')
                np.savez_compressed(QA/f'{name}.npz',rgb=rgb,valid=valid)
                big=Image.fromarray(rgb.astype(np.uint8)).resize((768,768))
                dr=ImageDraw.Draw(big)
                for iy,y in enumerate(range(32,384,64)):
                    for ix,x in enumerate(range(32,384,64)):
                        ident=iy*6+ix+1
                        dr.ellipse((2*x-4,2*y-4,2*x+4,2*y+4),outline='red',width=2)
                        dr.text((2*x+6,2*y-17),str(ident),fill='yellow',stroke_width=1,stroke_fill='black')
                        meta.append({'patch_id':name,'year':yr,'field':field,'phase':phase,'plot_id':int(row.plot_id),'N':n if False else int(row.N_kg_ha),'DAS':das,'point':ident,'x':x,'y':y,'role':'source_calibration' if (yr==2024 and field=='F1') else 'transfer_check','reference_label':''})
                tiles.append((name,big))
            panel=Image.new('RGB',(3*790,820),'white');dr=ImageDraw.Draw(panel)
            for k,(name,tile) in enumerate(tiles):panel.paste(tile,(k*790,42));dr.text((k*790+5,12),name,fill='black')
            panel.save(QA/f'points_{yr}_{field}_{phase}.png')
    pd.DataFrame(meta).to_csv(QA/'point_locations.csv',index=False)

def extract(df,gdf):
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];direct=df[df.slot_status=='AVAILABLE']
    for j,(rpath,sub) in enumerate(direct.groupby('raster_path'),1):
        with rasterio.open(rpath) as src:
            gg=gdf.to_crs(src.crs)
            for _,row in sub.iterrows():
                geom=gg[(gg.year==row.year)&(gg.field==row.field)&(gg.plot_id==int(row.plot_id))].geometry.iloc[0]
                rgb,valid=crop(src,geom)
                assert rgb[valid].max()<=255, 'Valid DN scale differs from registered 8-bit thresholds'
                r,g,b,sat,br=arrays(rgb);ms=rules(rgb,valid);n=int(valid.sum())
                d={c:row[c] for c in ['year','field','plot_id','block_design','N_kg_ha','DAS','flight_date','yield_kg_ha']}
                d.update(valid_pixels=n,roi_area_m2=float(gdf[(gdf.year==row.year)&(gdf.field==row.field)&(gdf.plot_id==int(row.plot_id))].geometry.iloc[0].area),raster_path=str(rpath),slot_status='DIRECT')
                for name,m in ms.items(): d[name]=float(m.sum()/n)
                for c,a in [('brightness',br),('saturation',sat)]:
                    q=np.percentile(a[valid],[10,25,50,75,90]);d.update({f'{c}_q{p}':float(v) for p,v in zip([10,25,50,75,90],q)})
                tot=r+g+b;exg=np.divide(2*g-r-b,tot,out=np.zeros_like(g),where=tot>0)
                d['exg_all_valid_mean']=float(exg[valid].mean())
                d['exg_positive_fraction']=float(((exg>0)&valid).sum()/n)
                # 4x4 equal-grid occupancy dispersion; map pattern, not boll count.
                for nm in ['green_canopeo','white_b140_s30','white_b160_s30']:
                    m=ms[nm];vals=[]
                    for ys in np.array_split(np.arange(m.shape[0]),4):
                        for xs in np.array_split(np.arange(m.shape[1]),4):
                            vv=valid[np.ix_(ys,xs)];mm=m[np.ix_(ys,xs)]
                            if vv.any():vals.append(mm.sum()/vv.sum())
                    d[nm+'_spatial_sd']=float(np.std(vals))
                rows.append(d)
        print(f'EXTRACT {j}/33 {Path(rpath).name} slots={len(rows)}',flush=True)
    out=pd.DataFrame(rows).sort_values(['year','field','plot_id','DAS'])
    assert len(out)==699 and not out.duplicated(['year','field','plot_id','DAS']).any()
    assert out.valid_pixels.min()>0
    out.to_csv(OUT/'direct_image_features.csv',index=False,float_format='%.10g')
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review-only',action='store_true');ap.add_argument('--extract-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(INPUT);gdf=gpd.read_file(GPKG,layer='inward_buffer_plot')
    if not args.extract_only:prepare_review(df,gdf)
    if args.review_only:return
    before=sha(INPUT)
    raw={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (ROOT/'01_Data/raw').rglob('*') if p.is_file() and p.name!='.DS_Store'}
    data=extract(df,gdf)
    assert sha(INPUT)==before
    after={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (ROOT/'01_Data/raw').rglob('*') if p.is_file() and p.name!='.DS_Store'}
    assert raw==after
    receipt={'status':'PASS','input_sha256':before,'geometry_sha256':sha(GPKG),'direct_slots':len(data),'raw_unchanged_metadata':raw==after,'raw_file_count':len(raw),'reconstructed_slots_excluded':15,'denominator':'pixels inside inward ROI with RGB sum>0, no RGB band equal to declared NoData, positive fourth band when present','harvest_confirmation':'User confirmed 2026-10-25: neither season had begun actual cotton picking at the 13 October flights.','observer':'Codex visual interpretation; no independent human annotation claimed'}
    (OUT/'extraction_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
