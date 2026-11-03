"""Train retrieval networks on fixed features; select epochs without test access.

The inner validation set selects an epoch count; a fresh model is then fitted on
all original training rows for that many epochs. Test is evaluated once per run.
This is explicitly a local protocol, not a claim of exact paper reproduction.
"""
import argparse, ast, hashlib, importlib.util, json, pickle, random, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
from common.metrics import label_relevance, map_all, recall_at_k

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def module(path, name):
    spec=importlib.util.spec_from_file_location(name, path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        allowed={('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),
                 ('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','scalar'),
                 ('numpy._core.multiarray','scalar'),('numpy.core.numeric','_frombuffer'),
                 ('numpy._core.numeric','_frombuffer')}
        if (mod,name) not in allowed: raise pickle.UnpicklingError(f'Unsupported data object: {mod}.{name}')
        return super().find_class(mod,name)

def load_data(method, path):
    hashes={}
    def sha(p): hashes[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    out={}
    if method=='DSCMR':
        for split in ('train','test'):
            for kind, field in [('image','img'),('text','txt'),('label','img_lab')]:
                p=path/f'{split}_{field}.mat'; sha(p)
                arr=loadmat(p)[f'{split}_{field}']
                out[f'{split}_{kind}']=arr.reshape(-1) if kind=='label' else arr
    else:
        for split in ('train','test'):
            p=path/f'clip_{split}.pkl'
            if not p.exists(): raise FileNotFoundError(f'Missing feature file: {p}')
            sha(p)
            with p.open('rb') as f: data=ArrayUnpickler(f).load()
            for kind in ('image','text','label'):
                arr=np.asarray(data[kind]); out[f'{split}_{kind}']=arr.reshape(-1) if kind=='label' else arr.squeeze()
            if 'ids' in data: out[f'{split}_ids']=np.asarray(data['ids']).reshape(-1).astype(str)
        if 'train_ids' in out and 'test_ids' in out:
            assert not set(out['train_ids']) & set(out['test_ids']), 'Train/test ID overlap'
    classes=np.unique(out['train_label'])
    mapping={v:i for i,v in enumerate(classes.tolist())}
    for split in ('train','test'):
        out[f'{split}_label']=np.array([mapping[v] for v in out[f'{split}_label']],dtype=np.int64)
        for kind in ('image','text'):
            x=np.asarray(out[f'{split}_{kind}'],dtype=np.float32)
            assert x.ndim==2 and len(x)==len(out[f'{split}_label']) and np.isfinite(x).all()
            out[f'{split}_{kind}']=x
    return out,hashes,len(classes)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False

def prototype_loss():
    # Upstream metrics.py has an unclosed docstring in unrelated Proxy_NCA.
    # Compile only the unchanged PAN definition used by its PCL branch.
    source=(ROOT/'CLIP4CMR/upstream/metrics.py').read_text(encoding='utf-8')
    body=source.split('def PAN(',1)[1].split('\ndef ',1)[0]
    namespace={'torch':torch,'F':torch.nn.functional}
    exec(compile('def PAN('+body,'upstream_PAN','exec'),namespace)
    return namespace['PAN']

def network(method, nc, di, dt):
    upstream=ROOT/method/'upstream'
    # Remove unused imports only. Network class bodies are verbatim upstream AST.
    tree=ast.parse((upstream/'model.py').read_text(encoding='utf-8'))
    keep=[]
    for node in tree.body:
        if isinstance(node, ast.Import):
            node.names=[n for n in node.names if n.name in ('torch','torch.nn','torch.nn.functional','math')]
            if not node.names: continue
        elif isinstance(node, ast.ImportFrom): continue
        elif not isinstance(node,(ast.ClassDef,ast.FunctionDef)): continue
        keep.append(node)
    tree.body=keep
    namespace={'__name__':'selected_model'}
    if method=='DSCMR': namespace['torch']=torch
    exec(compile(tree, str(upstream/'model.py'),'exec'), namespace)
    if method=='DSCMR': return namespace['IDCM_NN'](img_input_dim=di,text_input_dim=dt,output_dim=nc)
    return namespace['model'](num_class=nc,img_dim=di,text_dim=dt,mid_dim=256,feature_dim=1024)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['DSCMR','CLIP4CMR'],required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-epochs',type=int)
    parser.add_argument('--refit-epochs',type=int,help='Fixed full-train budget; overrides validation-selected epoch count')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out',type=Path)
    a=parser.parse_args()
    if a.refit_epochs is not None and a.refit_epochs<1: parser.error('--refit-epochs must be positive')
    if a.max_epochs is not None and a.max_epochs<1: parser.error('--max-epochs must be positive')
    a.data=(a.data or ROOT/a.method/'upstream/data/pascal').resolve()
    a.out=(a.out or ROOT/a.method/'runs'/f'pascal_validation_refit_seed{a.seed}').resolve()
    if (a.out/'result.json').exists(): raise RuntimeError('Completed result exists; choose a new --out to preserve it')
    a.out.mkdir(parents=True,exist_ok=True)
    status=ROOT/a.method/'status.json'
    write_json(status,{'method':a.method,'state':'loading','run':str(a.out)})
    try: data,hashes,nc=load_data(a.method,a.data)
    except Exception as e:
        write_json(status,{'method':a.method,'state':'blocked_missing_data','error':str(e),'run':str(a.out)})
        raise
    torch.set_num_threads(4)
    if a.device=='cuda': torch.cuda.reset_peak_memory_stats()
    epochs=a.max_epochs or (500 if a.method=='DSCMR' else 200)
    batch=100 if a.method=='DSCMR' else 300
    seed_all(a.seed)
    rng=np.random.default_rng(a.seed); inner_train=[]; val=[]
    for c in range(nc):
        ids=np.flatnonzero(data['train_label']==c); rng.shuffle(ids)
        nv=max(1,int(round(len(ids)*0.1)))
        assert len(ids)>nv
        val.extend(ids[:nv]);inner_train.extend(ids[nv:])
    inner_train=np.sort(inner_train);val=np.sort(val)
    assert not set(inner_train)&set(val)
    n=len(data['train_label']);nt=len(data['test_label'])
    protocol={'method':a.method,'seed':a.seed,'dataset':'Pascal-Sentence',
      'train_pairs':n,'inner_train_pairs':len(inner_train),'validation_pairs':len(val),'test_pairs':nt,
      'inner_train_indices':inner_train.tolist(),'validation_indices':val.tolist(),
      'feature_dimensions':[data['train_image'].shape[1],data['train_text'].shape[1]],
      'selection':'best mean validation mAP, then fresh full-train refit for selected epoch count',
      'max_selection_epochs':epochs,'batch_size':batch,'lr':1e-4,'adam_betas':[0.5,0.999],
      'test_used_for_model_selection':False,'paper_comparability_verified':False,
      'survey_reference':{'I2T':.674,'T2I':.682} if a.method=='DSCMR' else {'I2T':.698,'T2I':.692},
      'deviations':['Local validation/refit selection differs from upstream test-based checkpoint selection'],
      'training_scope':'retrieval network on fixed author features; encoders not retrained',
      'data_sha256':hashes,'torch':torch.__version__,'device':a.device}
    protocol['fixed_refit_epochs']=a.refit_epochs
    if a.refit_epochs is not None:
        protocol['selection']='record best validation checkpoint; fresh full-train refit for a fixed epoch budget'
        protocol['deviations'].append('Full-train refit budget fixed by --refit-epochs; validation best epoch does not determine final training duration')
    if nt!=200: protocol['deviations'].append(f'Author bundled test has {nt} pairs; survey reference uses 200')
    if a.method=='CLIP4CMR': protocol['deviations'].append('PCL explicitly selected; upstream main default PAN is not a handled loss name')
    provenance_path=a.data/'provenance.json'
    if provenance_path.exists():
        provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
        protocol['data_provenance']=provenance
        protocol['deviations'].extend(provenance.get('deviations',[]))
        protocol['training_scope']='retrieval network on locally extracted frozen CLIP RN50 features; encoders not retrained'
    write_json(a.out/'protocol.json',protocol)
    tr_x=torch.from_numpy(data['train_image']).to(a.device)
    tr_t=torch.from_numpy(data['train_text']).to(a.device)
    tr_y=torch.from_numpy(data['train_label']).to(a.device)
    onehot=torch.nn.functional.one_hot(tr_y,nc).float()
    if a.method=='DSCMR':
        # Read only loss definition, avoiding upstream training loop and its test selection.
        tree=ast.parse((ROOT/a.method/'upstream/train_model.py').read_text())
        tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('calc_loss','calc_label_sim')]
        ns={'torch':torch};exec(compile(tree,'upstream_loss','exec'),ns);loss_fn=ns['calc_loss']
    else: loss_fn=prototype_loss()
    def new_model():
        seed_all(a.seed)
        model=network(a.method,nc,tr_x.shape[1],tr_t.shape[1]).to(a.device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.5,.999))
        return model,opt
    def step(model,opt,indices):
        model.train(); loss_sum=0.
        order=np.array(indices,copy=True)
        if a.method=='CLIP4CMR': np.random.shuffle(order)
        for offset in range(0,len(order),batch):
            ix=torch.tensor(order[offset:offset+batch],device=a.device)
            opt.zero_grad(set_to_none=True);out=model(tr_x[ix],tr_t[ix])
            if a.method=='DSCMR': loss=loss_fn(*out,onehot[ix],onehot[ix],1e-3,1e-1)
            else:
                centers,fi,ft,_,_=out
                loss=loss_fn(fi,centers.T,tr_y[ix])+loss_fn(ft,centers.T,tr_y[ix])
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward();opt.step();loss_sum+=loss.item()*len(ix)
        return loss_sum/len(order)
    def evaluate(model,x,t,y):
        model.eval()
        with torch.inference_mode():
            out=model(x,t)
            fi,ft=out[:2] if a.method=='DSCMR' else out[1:3]
            fi=fi.cpu().numpy();ft=ft.cpu().numpy()
        scores=-cdist(fi,ft,'cosine' if a.method=='DSCMR' else 'euclidean')
        relevant=label_relevance(y,y)
        metrics={d:{**map_all(s,r),**recall_at_k(s,r)} for d,s,r in [('I2T',scores,relevant),('T2I',scores.T,relevant.T)]}
        return metrics,fi,ft
    start=time.perf_counter();model,opt=new_model();best=-1.;chosen=0;history=[]
    for epoch in range(1,epochs+1):
        loss=step(model,opt,inner_train)
        metrics,_,_=evaluate(model,tr_x[val],tr_t[val],data['train_label'][val])
        mean=(metrics['I2T']['mAP@ALL']+metrics['T2I']['mAP@ALL'])/2
        history.append({'epoch':epoch,'train_loss':loss,'validation':metrics})
        if mean>best:
            best=mean;chosen=epoch
            torch.save({'state_dict':model.state_dict(),'epoch':epoch},a.out/'validation_best.pt')
        if epoch==1 or epoch%50==0 or epoch==epochs:
            write_json(a.out/'history.json',history)
            write_json(status,{'method':a.method,'state':'training_validation','epoch':epoch,'total_epochs':epochs,'best_validation_epoch':chosen,'elapsed_seconds':time.perf_counter()-start})
            print(a.method,'validation',epoch,'loss',round(loss,6),'mAP',round(mean,5),'best_epoch',chosen,flush=True)
    del model,opt
    model,opt=new_model();refit=[]
    refit_epochs=a.refit_epochs if a.refit_epochs is not None else chosen
    for epoch in range(1,refit_epochs+1):
        loss=step(model,opt,np.arange(n));refit.append({'epoch':epoch,'train_loss':loss})
        if epoch==1 or epoch%50==0 or epoch==refit_epochs:
            write_json(status,{'method':a.method,'state':'refit_full_train','epoch':epoch,'total_epochs':refit_epochs,'elapsed_seconds':time.perf_counter()-start})
            write_json(a.out/'refit_history.json',refit)
            print(a.method,'refit',epoch,'/',refit_epochs,flush=True)
    write_json(a.out/'refit_history.json',refit)
    torch.save({'state_dict':model.state_dict(),'protocol':protocol,'best_validation_epoch':chosen,'training_epochs':refit_epochs},a.out/'model.pt')
    tx=torch.from_numpy(data['test_image']).to(a.device);tt=torch.from_numpy(data['test_text']).to(a.device)
    metrics,fi,ft=evaluate(model,tx,tt,data['test_label'])
    np.savez_compressed(a.out/'test_features.npz',image=fi,text=ft,label=data['test_label'])
    # Verify custom mAP independently against upstream evaluator, in both directions.
    if a.method=='DSCMR': ev=module(ROOT/a.method/'upstream/evaluate.py','upstream_eval')
    else:
        src=(ROOT/a.method/'upstream/evaluate.py').read_text()
        tree=ast.parse(src if src.lstrip().startswith('"""') else '"""'+src)
        tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='fx_calc_map_label']
        import scipy.spatial
        ns={'np':np,'scipy':sys.modules['scipy']};exec(compile(tree,'upstream_eval','exec'),ns)
        ev=type('Eval',(),{'fx_calc_map_label':staticmethod(ns['fx_calc_map_label'])})
    verified={}
    for direction,q,g in [('I2T',fi,ft),('T2I',ft,fi)]:
        value=float(ev.fx_calc_map_label(q,g,data['test_label']))
        assert abs(value-metrics[direction]['mAP@ALL'])<1e-8
        verified[direction]=value
    write_json(a.out/'verification.json',{'status':'passed','upstream_mAP':verified,'test_used_for_selection':False})
    result={'method':a.method,'status':'completed_local_protocol','dataset':'Pascal-Sentence',
      'metrics':metrics,'train_pairs':n,'test_pairs':nt,'validation_selection_epochs':epochs,
      'best_validation_epoch':chosen,'final_training_epochs':refit_epochs,'total_optimizer_epochs':epochs+refit_epochs,'seed':a.seed,
      'training_scope':protocol['training_scope'],'test_used_for_model_selection':False,
      'paper_comparability_verified':False,'deviations':protocol['deviations'],
      'elapsed_seconds':time.perf_counter()-start,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None}
    write_json(a.out/'result.json',result)
    write_json(status,{'method':a.method,'state':'completed_local_protocol','result':str(a.out/'result.json'),**result})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
