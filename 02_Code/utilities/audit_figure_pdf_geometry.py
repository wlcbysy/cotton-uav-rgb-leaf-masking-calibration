"""Poppler word-bbox audit, with PDF text directions and script font sizes."""
from pathlib import Path
import subprocess,xml.etree.ElementTree as ET,csv,argparse,json
import fitz

def audit(folder,output, *, max_height_mm=210.01, minimum_font_pt=6):
    rows=[];details=[]
    for p in sorted(Path(folder).glob('Figure_*.pdf')):
        svg=p.with_suffix('.svg');svg_root=ET.parse(svg).getroot()
        underscores=sum(''.join(t.itertext()).count('_') for t in svg_root.iter() if t.tag.split('}')[-1]=='text')
        xml=subprocess.check_output(['pdftotext','-bbox',str(p),'-'],stderr=subprocess.DEVNULL)
        root=ET.fromstring(xml);pdf=fitz.open(p)
        for idx,page in enumerate(root.iter('{http://www.w3.org/1999/xhtml}page')):
            spans=[]
            for b in pdf[idx].get_text('dict')['blocks']:
                for line in b.get('lines',[]):
                    for s in line['spans']:spans.append((fitz.Rect(s['bbox']),line['dir'],s['size'],s['text']))
            body=[];scripts=[];bad=[];font_body=[];font_script=[]
            for w in page.iter('{http://www.w3.org/1999/xhtml}word'):
                box=fitz.Rect(*[float(w.get(k)) for k in ('xMin','yMin','xMax','yMax')])
                best=max(spans,key=lambda s:(box&s[0]).get_area())
                vertical=abs(best[1][1])>.5;size=best[2];text=w.text or ''
                # Math superscripts/subscripts are identified by nominal PDF run size,
                # not by a short rotated glyph's vertical extent.
                script=size<6.99
                height=box.width if vertical else box.height
                (scripts if script else body).append(height);(font_script if script else font_body).append(size)
                limit=max(minimum_font_pt,6 if script else 7)
                if height<limit-.05 or size<limit-.05:bad.append({'text':text,'crossline_bbox_pt':round(height,3),'font_pt':round(size,3),'script':script,'vertical':vertical})
            width=float(page.get('width'))*25.4/72;height_mm=float(page.get('height'))*25.4/72
            row=dict(file=p.name,width_mm=round(width,3),height_mm=round(height_mm,3),svg_text_underscores=underscores,min_body_bbox_pt=round(min(body),3),min_script_bbox_pt=round(min(scripts),3) if scripts else '',min_body_font_pt=round(min(font_body),3),min_script_font_pt=round(min(font_script),3) if font_script else '',violations=len(bad),status='PASS' if width<=190.01 and height_mm<=max_height_mm and underscores==0 and not bad else 'FAIL')
            rows.append(row);details.append({'file':p.name,'violations':bad})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    output.with_suffix('.json').write_text(json.dumps({'method':'pdftotext -bbox words; PyMuPDF font sizes and text direction distinguish scripts/rotated axes. Cross-line dimension is used for rotated words. Raster/outlined standard-map labels are outside automated text-size scope; station and scale-bar labels are included. SVG underscores count only visible text nodes, not IDs or math-source comments.','pages':details},indent=2))
    print('\n'.join(str(r) for r in rows))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder');a.add_argument('output');v=a.parse_args();audit(v.folder,v.output)
