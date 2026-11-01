"""Preserve code, records and source history before removing retired binaries."""
from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/transition_20261014';OUT.mkdir(parents=True,exist_ok=True)
targets=['CLIP','ViLT','data/flickr30k','checkpoints/vilt_200k_mlm_itm.ckpt',
 'checkpoints/vilt_irtr_f30k.ckpt','checkpoints/ViT-B-32.pt','checkpoints/bert-base-uncased']
manifest=[]
for name in targets:
 p=(ROOT/name).resolve()
 if not p.is_relative_to(ROOT):raise ValueError(name)
 files=list(p.rglob('*')) if p.is_dir() else [p]
 files=[f for f in files if f.is_file()]
 manifest.append({'path':str(p),'files':len(files),'bytes':sum(f.stat().st_size for f in files)})
zpath=OUT/'retired_code_and_records.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for name in ('CLIP','ViLT','scripts','docs'):
  for p in (ROOT/name).rglob('*'):
   if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix.lower() in {'.py','.md','.json','.csv','.log','.txt','.patch','.cmd','.ps1'}:
    z.write(p,p.relative_to(ROOT))
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix.lower() in {'.md','.json','.csv','.cmd'}:z.write(p,p.name)
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None
 assert 'CLIP/runs/flickr30k_test_seed42_truncated/result.json' in z.namelist()
 assert 'ViLT/runs/flickr30k_author_test/result.json' in z.namelist()
record={'targets':manifest,'archive':str(zpath),'archive_sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),
 'archive_crc_verified':True,'reason':'User selected DSCMR and CLIP4CMR; retire Flickr30K evaluation pipelines',
 'kept':['.git','.venv310','sources','common','tests','DSCMR','CLIP4CMR','historical reports']}
(OUT/'cleanup_plan.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
