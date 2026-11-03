"""Negative result: QAP + cross-fitted ridge matching of DSCMR VGG features to UIUC images. See DSCMR/data_ids/NOTE.md."""
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from scripts.dscmr import loss_ablation as r1
from scripts.clip4cmr import improve as ci
from scipy.optimize import quadratic_assignment, linear_sum_assignment
data=ROOT/'DSCMR/upstream/data/pascal'
x,_,y,c=r1.load_split(data,'train'); xt,_,yt,_=r1.load_split(data,'test',c)
d=np.concatenate([x,xt]); dy=np.concatenate([y,yt])
ci.set_feats(ROOT/'CLIP4CMR/data/pascal_uiuc_seed42_multicaption')
L=np.load(ROOT/'CLIP4CMR/data/pascal_uiuc_seed42_multicaption/vitl14.npz'); u=L['image']; uy=L['labels']
n=lambda a:a/np.linalg.norm(a,axis=1,keepdims=True)
dn,un=n(d),n(u)
# global RSA-ish check: per class, QAP between similarity matrices, report objective vs random
rng=np.random.default_rng(0); res=[]
for k in range(20):
    r=np.flatnonzero(dy==k); cc=np.flatnonzero(uy==k)
    A=dn[r]@dn[r].T; B=un[cc]@un[cc].T
    A=A-A.mean(); B=B-B.mean()
    Ap=np.zeros((len(cc),len(cc))); Ap[:len(r),:len(r)]=A
    best=None
    for s in range(5):
        q=quadratic_assignment(Ap,B,method='faq',options={'maximize':True,'P0':'randomized','rng':s})
        if best is None or q.fun>best.fun: best=q
    perm=best.col_ind[:len(r)]
    # correlation of matched similarity entries
    iu=np.triu_indices(len(r),1)
    corr=np.corrcoef(A[iu], B[np.ix_(perm,perm)][iu])[0,1]
    rp=rng.permutation(len(cc))[:len(r)]
    corr0=np.corrcoef(A[iu], B[np.ix_(rp,rp)][iu])[0,1]
    res.append((k,round(corr,3),round(corr0,3)))
print(res)

# ---- iterative cross-fitted refinement ----
def qap_match():
    m=np.full(len(d),-1)
    for k in range(20):
        r=np.flatnonzero(dy==k); cc=np.flatnonzero(uy==k)
        A=dn[r]@dn[r].T; B=un[cc]@un[cc].T; A=A-A.mean(); B=B-B.mean()
        Ap=np.zeros((len(cc),len(cc))); Ap[:len(r),:len(r)]=A
        best=None
        for s in range(8):
            q=quadratic_assignment(Ap,B,method='faq',options={'maximize':True,'P0':'randomized','rng':np.random.default_rng(s)})
            if best is None or q.fun>best.fun: best=q
        m[r]=cc[best.col_ind[:len(r)]]
    return m
def ridge(X,Y,lam=10.):
    return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]), X.T@Y)
Z=dn-dn.mean(0)  # centred DSCMR features
def crossfit(m, folds=4):
    out=np.full(len(d),-1); marg=np.zeros(len(d))
    cls=np.arange(20); grp=cls%folds
    for g in range(folds):
        tr=np.isin(dy,cls[grp!=g]); te=~tr
        W=ridge(Z[tr], un[m[tr]]); P=n(Z[te]@W)
        for k in cls[grp==g]:
            r=np.flatnonzero(dy==k); cc=np.flatnonzero(uy==k)
            S=P[np.searchsorted(np.flatnonzero(te),r)]@un[cc].T
            ri,ci2=linear_sum_assignment(-S); out[r[ri]]=cc[ci2]
            s=np.sort(S,1); marg[r]=s[:,-1]-s[:,-2]
    return out, marg
m=qap_match()
for it in range(8):
    m2,marg=crossfit(m)
    print('iter',it,'agree prev',round((m2==m).mean(),3),'median margin',round(float(np.median(marg)),4),flush=True)
    if (m2==m).all(): break
    m=m2
np.save(ROOT/'DSCMR/data_ids/candidate_match.npy', m)
