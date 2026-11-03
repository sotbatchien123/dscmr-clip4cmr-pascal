"""Recover Pascal from UIUC and build a documented LOCAL CLIP4CMR split."""
import concurrent.futures,hashlib,json,pickle,time,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'CLIP4CMR/data/pascal_uiuc_seed42'
OUT.mkdir(parents=True,exist_ok=True)
records=json.loads((ROOT/'reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json').read_text(encoding='utf-8'))
def fetch(r):
    p=OUT/'images'/f"{r['id']}.jpg";p.parent.mkdir(exist_ok=True)
    for attempt in range(3):
        try:
            if not p.exists():
                with urllib.request.urlopen(r['image_url'],timeout=40) as response:
                    data=response.read()
                p.write_bytes(data)
            with Image.open(p) as im: im.verify()
            return {'id':r['id'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        except Exception:
            if p.exists(): p.unlink()
            if attempt==2: raise
            time.sleep(1)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        hashes=[]
        for i,h in enumerate(pool.map(fetch,records),1):
            hashes.append(h)
            if i%100==0: print('Verified images',i,'/1000',flush=True)
    import torch,clip
    torch.set_num_threads(4)
    model,preprocess=clip.load('RN50',device='cuda',download_root=str(ROOT/'checkpoints'))
    model.eval()
    images=[];texts=[]
    for start in range(0,len(records),32):
        batch=records[start:start+32]
        # One document per image, all available captions in source order.
        # Truncation is explicit; never silently discard images or split captions across sets.
        tokens=clip.tokenize([' '.join(r['captions']) for r in batch],truncate=True).cuda()
        pixels=torch.stack([preprocess(Image.open(OUT/'images'/f"{r['id']}.jpg").convert('RGB')) for r in batch]).cuda()
        with torch.inference_mode():
            images.append(model.encode_image(pixels).float().cpu().numpy())
            texts.append(model.encode_text(tokens).float().cpu().numpy())
        print('Encoded',min(start+32,1000),'/1000',flush=True)
    images=np.concatenate(images);texts=np.concatenate(texts)
    classes=sorted({r['class'] for r in records});labels=np.array([classes.index(r['class']) for r in records])
    ids=np.array([r['id'] for r in records]);rng=np.random.default_rng(42)
    train=[];test=[]
    for c in range(20):
        ix=np.flatnonzero(labels==c);rng.shuffle(ix);train.extend(ix[:40]);test.extend(ix[40:])
    train=np.sort(train);test=np.sort(test)
    assert len(train)==800 and len(test)==200 and not set(ids[train])&set(ids[test])
    for split,ix in [('train',train),('test',test)]:
        with (OUT/f'clip_{split}.pkl').open('wb') as f:
            pickle.dump({'image':images[ix],'text':texts[ix],'label':labels[ix],'ids':ids[ix]},f,protocol=4)
    provenance={'origin':'locally_extracted_UIUC','source':'https://vision.cs.uiuc.edu/pascal-sentences/',
      'split':'stratified per class 40 train / 10 test, numpy default_rng seed42',
      'train_ids':ids[train].tolist(),'test_ids':ids[test].tolist(),'class_names':classes,
      'text_rule':'join ALL available captions in page order with spaces; CLIP tokenize truncate=True (77 tokens)',
      'backbone':'OpenAI CLIP RN50 frozen','checkpoint_sha256':hashlib.sha256((ROOT/'checkpoints/RN50.pt').read_bytes()).hexdigest(),
      'image_hashes':hashes,'author_split_verified':False,'author_caption_preprocessing_verified':False,
      'deviations':['New local 800/200 split, not verified as author split','All captions concatenated and truncated at 77 tokens; differs from unverified author text preprocessing'],
      'feature_shapes':[list(images.shape),list(texts.shape)]}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('READY',OUT,flush=True)
