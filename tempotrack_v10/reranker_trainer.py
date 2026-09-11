"""Base-only, non-Test training for the V9.3 diagnostic pilot."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ..models.query_conditioned_reranker import CandidateReranker, FEATURE_NAMES, group_ranking_loss


def train(features_dir, output, *, device="cpu", epochs=12, seed=0):
    from ..orchestration.v9_oracle import sha256,write_json,memory_guard
    root=Path(features_dir); out=Path(output)
    meta=json.loads((root/"features.json").read_text())
    if meta["split"] not in ("train","val","dev") or meta["split"].lower().startswith("test"):
        raise ValueError("Test features are forbidden for optimizer and normalization")
    if not meta.get("base_only_supervision") or meta["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("unverified supervision/feature schema")
    for name,digest in meta["array_hashes"].items():
        if sha256(root/(name+".npy")) != digest: raise ValueError(f"feature hash mismatch: {name}")
    x=np.load(root/"features.npy",mmap_mode="r"); labels=np.load(root/"labels.npy",mmap_mode="r")
    allowed=np.load(root/"supervision_allowed.npy",mmap_mode="r")
    offsets=np.load(root/"offsets.npy"); videos=np.load(root/"videos.npy")
    train_groups=[]; holdout=[]
    for g,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        mask=allowed[a:b]; y=labels[a:b]
        if not ((mask&(y==1)).any() and (mask&(y==0)).any()): continue
        key=int(hashlib.sha256(str(int(videos[g])).encode()).hexdigest()[:8],16)%5
        (holdout if key==0 else train_groups).append(g)
    if not train_groups or not holdout: raise ValueError("need disjoint nonempty training/holdout videos")
    out.mkdir(parents=True,exist_ok=False); torch.manual_seed(seed); rng=np.random.default_rng(seed)
    model=CandidateReranker().to(device)
    # Normalization is fit only on supervised rows from training videos.
    total=np.zeros(x.shape[1],np.float64); squared=total.copy(); n=0
    for g in train_groups:
        a,b=offsets[g:g+2]; values=np.asarray(x[a:b][allowed[a:b]],dtype=np.float64)
        total+=values.sum(axis=0); squared+=(values*values).sum(axis=0); n+=len(values)
    mean=total/n; scale=np.sqrt(np.maximum(squared/n-mean*mean,1e-6))
    model.feature_mean.copy_(torch.tensor(mean,dtype=torch.float32,device=device))
    model.feature_scale.copy_(torch.tensor(scale,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)

    def batch(groups,hard=False):
        examples=[]; ys=[]
        for g in groups:
            a,b=offsets[g:g+2]; eligible=np.flatnonzero(allowed[a:b]); y=np.asarray(labels[a:b])
            if hard:
                pos=eligible[y[eligible]==1]; neg=eligible[y[eligible]==0]
                # Alternate prefilter and raw-support hard negatives; no random-negative curriculum.
                by_rank=sorted(neg,key=lambda i:(float(x[a+i,17]),int(i)))
                by_raw=sorted(neg,key=lambda i:(-float(x[a+i,5]),int(i)))
                selected=[]
                for u,v in zip(by_rank,by_raw):
                    for i in (u,v):
                        if i not in selected: selected.append(i)
                    if len(selected)>=16: break
                eligible=np.concatenate((pos,np.asarray(selected[:16],dtype=np.int64)))
            examples.append(np.asarray(x[a:b][eligible])); ys.append(y[eligible])
        width=max(map(len,ys)); bx=np.zeros((len(ys),width,x.shape[1]),np.float32); by=np.full((len(ys),width),-1,np.int64)
        for i,(features,y) in enumerate(zip(examples,ys)): bx[i,:len(y)]=features; by[i,:len(y)]=y
        return torch.from_numpy(bx).to(device),torch.from_numpy(by).to(device)

    def validation():
        model.eval(); losses=[]; correct=0; count=0
        with torch.inference_mode():
            for start in range(0,len(holdout),32):
                bx,by=batch(holdout[start:start+32]); z=model(bx); losses.append(float(group_ranking_loss(z,by)))
                winner=z.masked_fill(by<0,-torch.inf).argmax(dim=1); correct+=int((by.gather(1,winner[:,None])==1).sum()); count+=len(by)
        return float(np.mean(losses)),correct/max(1,count)

    best=float("inf"); history=[]; optimizer_steps=0; best_step=0
    with (out/"metrics.jsonl").open("x") as log:
        for epoch in range(epochs):
            memory_guard(); model.train(); shuffled=rng.permutation(train_groups); losses=[]
            for start in range(0,len(shuffled),32):
                bx,by=batch(shuffled[start:start+32],hard=True); opt.zero_grad(set_to_none=True)
                loss=group_ranking_loss(model(bx),by)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); optimizer_steps+=1; losses.append(float(loss.detach()))
            val,top1=validation(); row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),holdout_loss=val,holdout_top1=top1)
            history.append(row); log.write(json.dumps(row)+"\n"); log.flush(); print(json.dumps(row),flush=True)
            if val<best:
                best=val; best_step=optimizer_steps
                torch.save(dict(model_state=model.state_dict(),feature_names=FEATURE_NAMES,features_hash=sha256(root/"features.json"),
                    epoch=epoch+1,optimizer_steps=optimizer_steps,seed=seed,training_split=meta["split"],protocol=meta["protocol"],base_only_supervision=True,
                    test_weights_forbidden=True,feature_config=meta["feature_config"]),out/"best.pt")
    result=dict(status="COMPLETED",protocol=meta["protocol"],diagnostic_only=meta["protocol"]=="VAL_BASE_PILOT",
        paper_valid=meta["protocol"]!="VAL_BASE_PILOT",training_split=meta["split"],training_groups=len(train_groups),holdout_groups=len(holdout),
        train_video_ids=sorted({int(videos[g]) for g in train_groups}),holdout_video_ids=sorted({int(videos[g]) for g in holdout}),
        novel_gt_used=False,test_weights_used=False,base_only_supervision=True,normalization_fit="training-video supervised Base rows only",
        checkpoint=str(out/"best.pt"),checkpoint_hash=sha256(out/"best.pt"),features_hash=sha256(root/"features.json"),history=history,
        device=device,epochs=epochs,seed=seed,optimizer_steps=optimizer_steps,checkpoint_optimizer_steps=best_step,
        paper_status='NOT_PAPER_VALID' if meta['protocol']=='VAL_BASE_PILOT' else 'BASE_TRAIN',
        NOT_PAPER_VALID=meta['protocol']=='VAL_BASE_PILOT',
        environment=dict(python=sys.executable,python_version=platform.python_version(),torch_version=torch.__version__,
            variables={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_PRELOAD')}),
        repo_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        source_hashes={str(p):sha256(p) for p in (Path(__file__),Path(__file__).resolve().parents[1]/'models/query_conditioned_reranker.py',Path(__file__).resolve().parents[1]/'orchestration/v9_reranker.py')})
    write_json(out/"training.json",result)
    return result
