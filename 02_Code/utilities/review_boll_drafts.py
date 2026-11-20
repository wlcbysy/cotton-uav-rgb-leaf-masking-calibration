"""Local-only editor for the 21 AI drafts; human approval is always explicit."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import os
import threading
import webbrowser
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'00_Project/annotation'
WORK = DEST/'ai_review'
LOCK = threading.Lock()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,path)

def manifest():
    return json.loads((DEST/'ai_draft_manifest.json').read_text())['items']

def state():
    path=WORK/'review_state.json'
    return json.loads(path.read_text()) if path.exists() else {'items':{}}

def validate(name, lab):
    items={r['image_id']:r for r in manifest()}
    if name not in items: raise ValueError('未知图片')
    r=items[name]
    if digest(DEST/'images'/f'{name}.png') != r['source_sha256']:
        raise ValueError('原图 SHA-256 不匹配')
    valid=np.asarray(Image.open(DEST/'valid_masks'/f'{name}.png'))>0
    if lab.shape!=(384,384) or not set(np.unique(lab)) <= {0,1,255}:
        raise ValueError('标注必须为 384 × 384，像素值为 0 / 1 / 255')
    if np.any(lab[~valid]!=255): raise ValueError('无效区域必须保持未知')
    if not np.any(valid & (lab!=255)): raise ValueError('不能将全未知模板标为完成')
    return valid

class Handler(BaseHTTPRequestHandler):
    def reply(self, obj, code=200):
        data=json.dumps(obj,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

    def local_request(self):
        expected=f'127.0.0.1:{self.server.server_port}'
        if self.headers.get('Host')!=expected: raise ValueError('请使用启动器提供的本地地址')
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://{expected}': raise ValueError('不接受跨站请求')

    def do_GET(self):
        try:
            self.local_request()
            if self.path=='/':
                data=(DEST/'review_app.html').read_bytes()
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data);return
            if self.path=='/api/items':
                self.reply({'items':manifest(),'state':state()});return
            prefix='/api/item/'
            if self.path.startswith(prefix):
                name=self.path[len(prefix):]
                if name not in {r['image_id'] for r in manifest()}: raise ValueError('未知图片')
                path=WORK/'working_masks'/f'{name}.png'
                if not path.exists():path=DEST/'ai_draft_masks'/f'{name}.png'
                lab=np.asarray(Image.open(path));validate(name,lab)
                valid=(np.asarray(Image.open(DEST/'valid_masks'/f'{name}.png'))>0).astype('uint8')
                self.reply({'image':'data:image/png;base64,'+base64.b64encode((DEST/'images'/f'{name}.png').read_bytes()).decode(),
                            'labels':base64.b64encode(lab.tobytes()).decode(),
                            'valid':base64.b64encode(valid.tobytes()).decode(),
                            'record':state()['items'].get(name,{})});return
            self.reply({'error':'Not found'},404)
        except (ValueError,KeyError) as exc:self.reply({'error':str(exc)},400)

    def do_POST(self):
        try:
            self.local_request()
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<500000:raise ValueError('请求大小无效')
            data=json.loads(self.rfile.read(length))
            with LOCK:
                if self.path=='/api/save':
                    name=data['image_id'];raw=base64.b64decode(data['labels'],validate=True)
                    if len(raw)!=384*384:raise ValueError('标注尺寸错误')
                    lab=np.frombuffer(raw,dtype=np.uint8).reshape(384,384)
                    valid=validate(name,lab)
                    reviewer=str(data.get('reviewer','')).strip()
                    reviewed=data.get('reviewed') is True
                    if reviewed and (not reviewer or data.get('full_review') is not True):
                        raise ValueError('请填写真实审核人，并确认已检查整张切片')
                    path=WORK/'working_masks'/f'{name}.png';path.parent.mkdir(parents=True,exist_ok=True)
                    temp=path.with_suffix('.tmp.png');Image.fromarray(lab).save(temp);os.replace(temp,path)
                    s=state();s['items'][name]={'reviewed':reviewed,'reviewer':reviewer,
                        'notes':str(data.get('notes','')),'saved_at':now(),'mask_sha256':digest(path),
                        'unknown_fraction':float(((lab==255)&valid).sum()/valid.sum()),
                        'label_provenance':'AI-assisted draft corrected and reviewed by human' if reviewed else 'Unconfirmed working draft'}
                    # Editing a previously finalized image invalidates the declaration.
                    decl_path=DEST/'annotation_declaration.json'
                    decl=json.loads(decl_path.read_text())
                    if decl.get('human_completed'):
                        decl['human_completed']=False;decl['notes']='Review changed after finalization; finalize all images again.'
                        atomic_json(decl_path,decl)
                    s.pop('finalized_at',None);atomic_json(WORK/'review_state.json',s)
                    self.reply({'ok':True,'state':s});return
                if self.path=='/api/finalize':
                    reviewer=str(data.get('reviewer','')).strip()
                    if not reviewer or data.get('confirm') is not True:
                        raise ValueError('需要真实审核人和最终确认')
                    s=state();items=manifest()
                    checked=[]
                    for r in items:
                        name=r['image_id'];record=s['items'].get(name,{})
                        path=WORK/'working_masks'/f'{name}.png'
                        if not record.get('reviewed') or not record.get('reviewer'):
                            raise ValueError(f'尚未审核：{name}')
                        if not path.exists() or digest(path)!=record['mask_sha256']:
                            raise ValueError(f'审核后文件发生变化：{name}')
                        validate(name,np.asarray(Image.open(path)));checked.append((name,path))
                    (DEST/'completed_masks').mkdir(exist_ok=True)
                    for name,path in checked:
                        target=DEST/'completed_masks'/f'{name}.png'
                        temp=target.with_suffix('.tmp');temp.write_bytes(path.read_bytes());os.replace(temp,target)
                    s['finalized_at']=now();s['finalized_by']=reviewer
                    atomic_json(WORK/'review_state.json',s)
                    atomic_json(DEST/'annotation_declaration.json',{
                        'human_completed':True,'annotator':reviewer,
                        'label_provenance':'AI-assisted labels; full human review and correction',
                        'ai_assisted':True,'independent_unassisted_annotation':False,
                        'review_log':'ai_review/review_state.json','finalized_at':s['finalized_at'],
                        'notes':'All 21 crops explicitly confirmed by named human reviewers. Retain AI assistance and unknown fractions in reporting. No scoring triggered by this editor.'})
                    self.reply({'ok':True,'state':s,'message':'21 张标注已保存至 completed_masks。请通知 Codex 检查并评分；本页面未计算精度。'});return
                self.reply({'error':'Not found'},404)
        except (ValueError,KeyError,TypeError) as exc:self.reply({'error':str(exc)},400)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8765);ap.add_argument('--open',action='store_true');args=ap.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url=f'http://127.0.0.1:{server.server_port}/'
    print('开铃像素审核：'+url,flush=True)
    print('仅本机访问；Ctrl+C 关闭。所有写入仅限 annotation 目录。',flush=True)
    if args.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
