"""Verify saved results and CLIP4CMR implementation without inventing data."""
import json
import numpy as np
import torch
from scipy.spatial.distance import cdist
from train_selected import ROOT, network, module, write_json, prototype_loss
from common.metrics import map_all, label_relevance, recall_at_k

torch.set_num_threads(4)
run=ROOT/'DSCMR/runs/pascal_validation_refit_seed42'
result=json.loads((run/'result.json').read_text())
data=np.load(run/'test_features.npz')
relevant=label_relevance(data['label'],data['label'])
scores=-cdist(data['image'],data['text'],'cosine')
for direction,s in [('I2T',scores),('T2I',scores.T)]:
    for key,value in {**map_all(s,relevant),**recall_at_k(s,relevant)}.items():
        assert abs(value-result['metrics'][direction][key])<1e-10
# Synthetic tensors check forward/backward only; NOT a dataset experiment.
torch.manual_seed(42)
model=network('CLIP4CMR',20,1024,1024)
centers,fi,ft,*_=model(torch.randn(8,1024),torch.randn(8,1024))
loss_fn=prototype_loss()
y=torch.arange(8)
loss=loss_fn(fi,centers.T,y)+loss_fn(ft,centers.T,y)
assert torch.isfinite(loss)
loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
ev=module(ROOT/'CLIP4CMR/upstream/evaluate.py','clip4_eval')
labels=np.arange(8)//2
for q,g in [(fi,ft),(ft,fi)]:
    q=q.detach().numpy();g=g.detach().numpy()
    actual=map_all(-cdist(q,g),label_relevance(labels,labels))['mAP@ALL']
    assert abs(actual-ev.fx_calc_map_label(q,g,labels))<1e-10
record={'status':'passed','DSCMR_saved_metrics_recomputed':True,
        'CLIP4CMR_synthetic_forward_backward_and_evaluator':True,
        'CLIP4CMR_real_experiment_completed':False}
write_json(ROOT/'reports/verification_selected.json',record)
print(json.dumps(record,indent=2))
