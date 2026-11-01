"""Verify the replacement inputs and recompute its measured retrieval metrics."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from train_selected import ROOT,load_data
from common.metrics import label_relevance,map_all,recall_at_k
directory=ROOT/'CLIP4CMR/data/pascal_uiuc_seed42'
data,hashes,nc=load_data('CLIP4CMR',directory)
assert nc==20
for split,n,count in [('train',800,40),('test',200,10)]:
    assert data[split+'_image'].shape==data[split+'_text'].shape==(n,1024)
    assert np.all(np.bincount(data[split+'_label'])==count)
p=json.loads((directory/'provenance.json').read_text())
assert set(p['train_ids'])==set(data['train_ids']) and set(p['test_ids'])==set(data['test_ids'])
image_hashes={x['id']:x['sha256'] for x in p['image_hashes']}
assert not {image_hashes[i] for i in p['train_ids']}&{image_hashes[i] for i in p['test_ids']}, 'Same image bytes across train/test'
assert p['checkpoint_sha256']=='afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
run=ROOT/'CLIP4CMR/runs/pascal_uiuc_seed42'
record={'input_verified':True,'same_image_bytes_across_train_test':False,'author_split_verified':False,'files_sha256':hashes}
if (run/'result.json').exists():
    result=json.loads((run/'result.json').read_text())
    z=np.load(run/'test_features.npz');rel=label_relevance(z['label'],z['label'])
    scores=-cdist(z['image'],z['text'],'euclidean')
    for direction,s in [('I2T',scores),('T2I',scores.T)]:
        for key,value in {**map_all(s,rel),**recall_at_k(s,rel)}.items():
            assert abs(value-result['metrics'][direction][key])<1e-10
    record['saved_test_metrics_recomputed']=True
(ROOT/'reports/verification_pascal_replacement.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
