"""V9.3 bounded reranker pilot. No Test optimizer, detector, or Novel selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import sqlite3
from types import SimpleNamespace

import numpy as np

from ..models.query_conditioned_reranker import (
    CandidateReranker, EVIDENCE_COLUMNS, FEATURE_NAMES, add_event_context, candidate_features,
)
from .v9_oracle import sha256, write_json, memory_guard


def self_check(output):
    import torch
    from ..models.query_conditioned_reranker import group_ranking_loss
    torch.manual_seed(0); rng=np.random.default_rng(0)
    cosine=rng.uniform(-1,1,(4,7)).astype(np.float32); evidence=rng.uniform(0,1,(7,7)).astype(np.float32)
    first=candidate_features(cosine,evidence,10,1)
    features=add_event_context([first,candidate_features(cosine*.9,evidence,15,2)])
    assert np.isfinite(features).all() and features.shape==(2,len(FEATURE_NAMES))
    model=CandidateReranker(); model.eval(); x=torch.tensor(features)
    a=model(x); b=torch.cat([model(x[i:i+1]) for i in range(len(x))]); assert torch.allclose(a,b,rtol=1e-5,atol=1e-6)
    # Multi-positive reference, masked unknown, and real backward gradients.
    z=torch.tensor([[.3,-.2,.6,100.]],requires_grad=True); y=torch.tensor([[1,0,1,-1]])
    loss=group_ranking_loss(z,y)
    expected=torch.logsumexp(z[0,:3],dim=0)-torch.logsumexp(z[0,[0,2]],dim=0)+.2*torch.relu(.2+z[0,1]-z[0,[0,2]].max())
    assert torch.allclose(loss,expected); loss.backward(); assert torch.isfinite(z.grad).all() and z.grad[0,3]==0
    bx=torch.tensor(np.stack([features,features*.9])); by=torch.tensor([[1,0],[0,1]])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001); optimizer.zero_grad(); real_loss=group_ranking_loss(model(bx),by); real_loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]; assert grads and all(torch.isfinite(g).all() for g in grads)
    magnitude=sum(float(g.abs().sum()) for g in grads); assert magnitude>0; optimizer.step()
    # Metadata is never an argument to the numerical feature/model interface.
    original=dict(label=1,target_base=1,candidate_base=1,group_id=7)
    changed=dict(label=-1,target_base=0,candidate_base=0,group_id=900)
    def from_metadata_agnostic_input(meta):
        return add_event_context([candidate_features(cosine,evidence,10,1),candidate_features(cosine*.9,evidence,15,2)])
    assert np.array_equal(from_metadata_agnostic_input(original),from_metadata_agnostic_input(changed))
    records=[dict(video_id=1,frame_index=i,image_id=i,track_id=10 if i==0 else 20,
        score=.9,_box_xyxy=[0,0,1,1],bbox=[0,0,1,1],category_id=1,observation_uid=str(i)) for i in range(5)]
    embeddings=np.tile(np.asarray([[1.,0.,0.]],np.float32),(5,1))
    config=dict(query_observations=4,top_r=3,min_gap=0,max_gap=360)
    events,fragments,_=native_video_scores(records,embeddings,model,config)
    assert len(events)==1 and events[0]['target']==1 and events[0]['decision_frame']==4
    changed_records=[{**r,'label':-1,'target_base':0,'candidate_base':0,'group_id':999} for r in records]
    repeated,_,_=native_video_scores(changed_records,embeddings,model,config)
    assert repeated==events
    sequential=[{**records[i],'track_id':10+i} for i in range(3)]
    rewritten,stats=replay_scores(sequential,[np.array([i]) for i in range(3)],
        [dict(target=1,decision_frame=1,candidates=[0],scores=[1.]),dict(target=2,decision_frame=2,candidates=[0],scores=[1.])],0.,0.)
    assert [r['track_id'] for r in rewritten]==[10,10,10] and stats.get('competition_reject',0)==0
    result=dict(status='PASSED',executed_checks=['finite_features','batched_single_score_equality','multipos_formal_loss',
        'unknown_label_zero_gradient','finite_nonzero_model_backward','optimizer_update','nonfeature_metadata_invariance',
        'B4_last_visible_query_decision_frame','native_scores_ignore_label_and_group_metadata','event_local_root_reuse'],
        multipos_loss=float(loss.detach()),reference_loss=float(expected.detach()),model_gradient_l1=magnitude,
        unknown_gradient=float(z.grad[0,3]),feature_names=list(FEATURE_NAMES),source_hashes=source_hashes())
    write_json(output,result); return result


def source_hashes():
    base=Path(__file__).resolve().parents[1]
    return {str(p):sha256(p) for p in (Path(__file__),base/'models/query_conditioned_reranker.py',base/'training/reranker_trainer.py')}


def native_video_scores(records, embeddings, model, config, *, capture_features=False):
    """All native fragments, no annotation, cached labels, or GT eligibility.

    Uses schema10's exact native-last-observation B-specific Top64 prefilter.
    Production event-frame root competition/collision is applied downstream.
    """
    import torch
    from ..streaming.psmr_dataset import fragment_rows
    from ..streaming.partial_support import build_memory_anchor
    from .v9_parameter_search import _rank_candidates
    frames=np.asarray([r['frame_index'] for r in records]); ids=np.asarray([r['track_id'] for r in records])
    rows=fragment_rows(ids,frames); boxes=np.asarray([r['_box_xyxy'] for r in records],dtype=np.float32); det=np.asarray([r['score'] for r in records],dtype=np.float32)
    video=int(records[0]['video_id']); anchors=[]; info=[]
    for serial,indices in enumerate(rows):
        anchors.append(build_memory_anchor(fragment_id=str(serial),root_id=int(ids[indices[0]]),video_id=video,rows=indices,
            features=embeddings,boxes_xyxy=boxes,scores=det,frames=frames,capacity=64,max_gap=config['max_gap']))
        info.append(dict(serial=serial,rows=indices,first=int(frames[indices[0]]),last=int(frames[indices[-1]])))
    by_last=defaultdict(list)
    for item in info: by_last[item['last']].append(item)
    events=[]; captured={}; device=next(model.parameters()).device; model.eval()
    for target in info:
        first=target['first']; legal=[c for f in range(first-config['max_gap'],first-config['min_gap']+1)
            for c in by_last.get(f,[]) if c['last']<first]
        if not legal: continue
        ranked=_rank_candidates(SimpleNamespace(features=embeddings),target,legal,query_count=config['query_observations'])[:64]
        # Cache rows are chronological candidate-last/serial, not rank order.
        # Preserve that ordering for deterministic tied raw-rank percentiles.
        ranked.sort(key=lambda item:(item[1]['last'],item[1]['serial']))
        qrows=target['rows'][:config['query_observations']]; q=embeddings[qrows]; q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-6)
        candidate_serials=[]; per_candidate=[]
        for rank,candidate in ranked:
            anchor=anchors[candidate['serial']]; m=anchor.features; m=m/np.maximum(np.linalg.norm(m,axis=1,keepdims=True),1e-6)
            per_candidate.append(candidate_features(q@m.T,anchor.evidence,first-candidate['last'],rank,top_r=config['top_r'],max_gap=config['max_gap']))
            candidate_serials.append(candidate['serial'])
        features=add_event_context(per_candidate)
        with torch.inference_mode(): scores=model(torch.from_numpy(features).to(device)).cpu().numpy()
        if not np.isfinite(scores).all(): raise FloatingPointError('nonfinite native inference score')
        events.append(dict(target=target['serial'],decision_frame=int(frames[qrows[-1]]),candidates=candidate_serials,scores=scores.tolist(),raw_support=features[:,5].tolist()))
        if capture_features: captured[target['serial']]=(candidate_serials,features)
    return events,rows,captured


def replay_scores(records, fragment_rows, events, threshold, margin):
    from ..streaming.partial_support import _frame_occupancy,_has_frame_collision,_apply_fragment_target
    result=[dict(r) for r in records]; occupancy=_frame_occupancy(result); roots={s:int(records[rows[0]]['track_id']) for s,rows in enumerate(fragment_rows)}
    groups=defaultdict(list); stats=Counter()
    for event in events: groups[event['decision_frame']].append(event)
    for frame,items in sorted(groups.items()):
        proposals=[]
        for event in items:
            order=sorted(range(len(event['scores'])),key=lambda i:(-event['scores'][i],event['candidates'][i])); i=order[0]
            score=event['scores'][i]; delta=score-event['scores'][order[1]] if len(order)>1 else float('inf')
            if score<threshold or delta<margin: continue
            c=event['candidates'][i]; proposals.append((event['target'],roots[c],score,delta,c))
        winners={}
        for proposal in sorted(proposals,key=lambda p:(-p[2],-p[3],p[0])):
            winners.setdefault(proposal[1],proposal)
        for t,root,score,delta,c in sorted(proposals):
            if winners[root][0]!=t: stats['competition_reject']+=1; continue
            fragment=SimpleNamespace(fragment_rows=fragment_rows[t]); source=roots[t]
            if root!=source and _has_frame_collision(occupancy,result,fragment,root): stats['frame_collision_reject']+=1; continue
            if root!=source:
                stats['changed_observations']+=_apply_fragment_target(occupancy,result,fragment,source,root); stats['changed_fragments']+=1
            roots[t]=root
    return result,dict(stats)


def load_checkpoint(checkpoint, device='cpu'):
    import torch
    cp=Path(checkpoint); receipt=json.loads((cp.parent/'training.json').read_text())
    if receipt['training_split']=='test' or receipt.get('test_weights_used') or not receipt['base_only_supervision']:
        raise ValueError('invalid optimizer provenance')
    if sha256(cp)!=receipt['checkpoint_hash']: raise ValueError('checkpoint hash mismatch')
    state=torch.load(cp,map_location=device)
    if state['training_split']=='test' or tuple(state['feature_names'])!=FEATURE_NAMES: raise ValueError('invalid checkpoint')
    model=CandidateReranker().to(device); model.load_state_dict(state['model_state']); model.eval()
    return model,state['feature_config'],receipt


def read_native_video(shard, source_by_uid):
    from ..v6_cli import _frames_for_shard,_native_uid
    if sha256(shard['path'])!=shard['sha256']: raise ValueError('native shard hash mismatch')
    records=[]; embeddings=[]
    for frame in _frames_for_shard(shard):
        for index in range(len(frame.scores)):
            uid=_native_uid(shard['video_id'],frame.frame_id,index); row=dict(source_by_uid.pop(uid))
            if row['frame_index']!=int(frame.frame_id) or row['image_id']!=int(frame.image_id): raise ValueError('native/source UID binding mismatch')
            row['_box_xyxy']=frame.boxes_xyxy[index].tolist(); records.append(row)
        embeddings.append(frame.embeddings_raw)
    if source_by_uid: raise ValueError('native cache failed to cover source observations')
    return records,np.concatenate(embeddings).astype(np.float32)


def native_parity(cache, features_dir, checkpoint, output, videos_limit=3):
    import ijson
    import torch
    meta=json.loads((Path(cache)/'metadata.json').read_text()); root=Path(features_dir)
    dataset=json.loads((root/'features.json').read_text()); offsets=np.load(root/'offsets.npy'); group_videos=np.load(root/'videos.npy')
    features=np.load(root/'features.npy',mmap_mode='r'); selected=set(map(int,np.unique(group_videos)[:videos_limit]))
    model,config,receipt=load_checkpoint(checkpoint); grouped={v:{} for v in selected}
    with Path(meta['frontend_prediction']).open('rb') as handle:
        for row in ijson.items(handle,'item',use_float=True):
            if row['video_id'] in selected: grouped[row['video_id']][row['observation_uid']]=row
    expected=defaultdict(dict); group_index=0
    with Path(meta['rows_path']).open() as handle:
        for key,rows in itertools.groupby((json.loads(line) for line in handle),lambda e:(e['video_id'],e['target_serial'])):
            keep=[e for e in rows if 1<=e[f'prefilter_rank_b{config["query_observations"]}']<=64]
            if not keep: continue
            if key[0] in selected:
                a,b=offsets[group_index:group_index+2]; expected[key[0]][key[1]]=([e['candidate_serial'] for e in keep],np.array(features[a:b]))
            group_index+=1
    native=json.loads(Path(meta['manifest']).read_text()); comparisons=0; max_feature=0.; max_score=0.
    for shard in native['shards']:
        video=int(shard['video_id'])
        if video not in selected: continue
        records,embeddings=read_native_video(shard,grouped[video]); events,fragments,actual=native_video_scores(records,embeddings,model,config,capture_features=True)
        for target,(candidate_ids,cached) in expected[video].items():
            actual_ids,values=actual[target]
            if actual_ids!=candidate_ids: raise AssertionError('cache/native candidate prefilter mismatch')
            np.testing.assert_allclose(values,cached,rtol=2e-5,atol=2e-6)
            with torch.inference_mode(): a=model(torch.from_numpy(values)); b=model(torch.from_numpy(cached))
            np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=2e-5,atol=2e-6)
            max_feature=max(max_feature,float(np.abs(values-cached).max())); max_score=max(max_score,float((a-b).abs().max())); comparisons+=len(values)
        for event in events:
            visible=fragments[event['target']][:config['query_observations']]
            assert event['decision_frame']==records[visible[-1]]['frame_index']
    if not comparisons: raise AssertionError('no real parity comparisons')
    result=dict(status='PASSED',artifact='v9_3_cache_native_inference_parity',videos=sorted(selected),candidate_comparisons=comparisons,
        max_feature_abs_error=max_feature,max_logit_abs_error=max_score,rtol=2e-5,atol=2e-6,checkpoint_hash=sha256(checkpoint),features_hash=sha256(root/'features.json'),
        source_hashes=source_hashes(),inference_eligibility='all native fragments; no GT target filter',query_clock='last visible B4 query row')
    write_json(output,result); return result


def _validate_score_shard(doc, video, source, shard, config):
    """Validate saved inference without running the model or rebuilding features."""
    from ..streaming.psmr_dataset import fragment_rows
    if doc['video_id'] != video:
        raise ValueError('saved score video mismatch')
    records, embeddings = read_native_video(shard, {uid: json.loads(payload) for _, uid, payload in source})
    del embeddings
    uids = [r['observation_uid'] for r in records]
    if doc['native_uids'] != uids:
        raise ValueError('saved native observation order mismatch')
    frames = np.asarray([r['frame_index'] for r in records])
    rows = fragment_rows(np.asarray([r['track_id'] for r in records]), frames)
    if doc['fragment_rows'] != [r.tolist() for r in rows]:
        raise ValueError('saved fragment/source row mismatch')
    first = [int(frames[r[0]]) for r in rows]; last = [int(frames[r[-1]]) for r in rows]
    expected = {t for t in range(len(rows)) if any(
        last[c] < first[t] and config['min_gap'] <= first[t]-last[c] <= config['max_gap']
        for c in range(len(rows)))}
    seen = set()
    for event in doc['events']:
        t = event['target']; candidates = event['candidates']
        if t not in expected or t in seen:
            raise ValueError('saved event target coverage mismatch')
        seen.add(t)
        if event['decision_frame'] != int(frames[rows[t][:config['query_observations']][-1]]):
            raise ValueError('saved B4 decision clock mismatch')
        legal = {c for c in range(len(rows)) if last[c] < first[t]
                 and config['min_gap'] <= first[t]-last[c] <= config['max_gap']}
        if (len(candidates) != min(64, len(legal)) or len(set(candidates)) != len(candidates)
                or not set(candidates) <= legal):
            raise ValueError('saved candidate legality/coverage mismatch')
        if candidates != sorted(candidates, key=lambda c: (last[c], c)):
            raise ValueError('saved candidate order mismatch')
        for key in ('scores', 'raw_support'):
            if len(event[key]) != len(candidates) or not np.isfinite(event[key]).all():
                raise ValueError('saved score length/nonfinite mismatch')
    if seen != expected:
        raise ValueError('saved event coverage incomplete')
    return len(records), len(doc['events'])


def score_dataset(cache, checkpoint, source_index, subset_manifest, output, *, resume=False, legacy_validation=None):
    """Read original frontend payloads only; oracle IDs/labels are never read."""
    import fcntl
    out=Path(output)
    # Old workers predate the advisory lock. Refuse a duplicate by exact argv,
    # not by an ambiguous process name or a potentially recycled PID.
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name)==os.getpid(): continue
        try: argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (OSError,PermissionError): continue
        if b'score-native' in argv and b'--output' in argv:
            pos=argv.index(b'--output')+1
            if pos<len(argv) and os.fsdecode(argv[pos])==str(out):
                raise RuntimeError(f'native scoring output already owned by PID {proc.name}')
    out.mkdir(parents=True,exist_ok=resume)
    lock=(out/'.score.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta=json.loads((Path(cache)/'metadata.json').read_text())
    for key in ('manifest','frontend_prediction','annotation'):
        if sha256(meta[key])!=meta[key+'_hash']: raise ValueError(f'{key} hash mismatch')
    source_index=Path(source_index); index_receipt=json.loads((source_index.parent/'oracle.json').read_text())
    if index_receipt['inputs']['frontend_prediction']['sha256']!=meta['frontend_prediction_hash']:
        raise ValueError('original payload index source mismatch')
    model,config,training=load_checkpoint(checkpoint)
    native=json.loads(Path(subset_manifest or meta['manifest']).read_text()); shards=sorted(native['shards'],key=lambda s:int(s['video_id']))
    if subset_manifest and len(shards)!=128: raise ValueError('selection requires exactly 128 native videos')
    binding=dict(checkpoint_hash=sha256(checkpoint),feature_config=config,
        source_prediction_hash=meta['frontend_prediction_hash'],event_cache_metadata_hash=sha256(Path(cache)/'metadata.json'),
        native_manifest_hash=sha256(subset_manifest or meta['manifest']),source_index_hash=sha256(source_index))
    receipt_path=out/'score_run_binding.json'; legacy=None
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if old['binding']!=binding: raise ValueError('resume run binding mismatch')
    elif resume and (out/'source.sqlite').exists():
        if not legacy_validation: raise ValueError('legacy partial requires explicit --legacy-validation provenance')
        legacy=json.loads(Path(legacy_validation).read_text())
        if legacy.get('status')!='PASSED' or legacy.get('checkpoint_hash')!=binding['checkpoint_hash']:
            raise ValueError('legacy checkpoint validation mismatch')
        snapshots=[v for k,v in legacy.get('worker_import_snapshots',{}).items() if k.startswith('native_scoring_pid_')]
        if len(snapshots)!=1: raise ValueError('legacy scoring worker provenance ambiguous')
        for name, digest in snapshots[0].items():
            if name.endswith(('query_conditioned_reranker.py','reranker_trainer.py')) and sha256(name)!=digest:
                raise ValueError('legacy model/trainer content changed')
    elif resume and any((out/'videos').glob('*.json')):
        raise ValueError('saved scores missing original source database')
    read_db=sqlite3.connect(f'file:{source_index}?mode=ro',uri=True); write_db=sqlite3.connect(out/'source.sqlite')
    write_db.execute('CREATE TABLE IF NOT EXISTS rows (ordinal INTEGER PRIMARY KEY, video INTEGER, uid TEXT UNIQUE, payload TEXT)')
    if write_db.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise ValueError('source SQLite integrity failure')
    (out/'videos').mkdir(exist_ok=resume); count=0; event_count=0; reused=[]; missing=[]; saved_hashes={}
    expected_names={f'{int(s["video_id"])}.json' for s in shards}
    if any(p.name not in expected_names for p in (out/'videos').glob('*.json')):
        raise ValueError('unexpected saved score video')
    if set(r[0] for r in write_db.execute('SELECT DISTINCT video FROM rows'))-set(int(s['video_id']) for s in shards):
        raise ValueError('unexpected source database video')
    # Audit all existing artifacts before computing even one missing video.
    for shard in shards:
        video=int(shard['video_id']); path=out/'videos'/f'{video}.json'
        source=list(read_db.execute('SELECT ordinal,uid,payload FROM rows WHERE video=? ORDER BY ordinal',(video,)))
        saved=list(write_db.execute('SELECT ordinal,uid,payload FROM rows WHERE video=? ORDER BY ordinal',(video,)))
        if saved and saved!=source: raise ValueError(f'saved original payload mismatch: video {video}')
        if path.exists():
            doc=json.loads(path.read_text()); n,e=_validate_score_shard(doc,video,source,shard,config)
            saved_hashes[path.name]=sha256(path); count+=n; event_count+=e; reused.append(video)
            # Atomic score rename may have preceded the original SQLite commit.
            # Restore absent original rows only; never replace existing rows.
            if not saved:
                write_db.executemany('INSERT INTO rows VALUES (?,?,?,?)',((o,video,u,p) for o,u,p in source)); write_db.commit()
        else: missing.append(video)
    if receipt_path.exists():
        for name,digest in old.get('adopted_score_hashes',{}).items():
            if saved_hashes.get(name)!=digest: raise ValueError('adopted score hash changed')
    else:
        write_json(receipt_path,dict(binding=binding,source_hashes=source_hashes(),adopted_score_hashes=saved_hashes,
            legacy_validation_path=str(legacy_validation) if legacy else None,
            legacy_validation_hash=sha256(legacy_validation) if legacy else None,
            provenance_note='Legacy worker snapshot plus structural/source audit; hashes first pinned at adoption, not retroactively at original spawn.' if legacy else 'fresh run'))
    if (out/'scores.json').exists():
        complete=json.loads((out/'scores.json').read_text())
        if missing or complete['score_file_hashes']!=saved_hashes or complete['checkpoint_hash']!=binding['checkpoint_hash']:
            raise ValueError('completed score receipt mismatch')
        write_db.close(); read_db.close(); lock.close(); return complete
    print(json.dumps(dict(stage='resume_audit',reused_videos=reused,missing_videos=missing,scoring_recomputed=0)),flush=True)
    for index,shard in enumerate(shards):
        memory_guard(); video=int(shard['video_id']); source=list(read_db.execute('SELECT ordinal,uid,payload FROM rows WHERE video=?',(video,)))
        if video in reused: continue
        by_uid={uid:json.loads(payload) for _,uid,payload in source}; records,embeddings=read_native_video(shard,by_uid)
        events,rows,_=native_video_scores(records,embeddings,model,config)
        write_json(out/'videos'/f'{video}.json',dict(video_id=video,events=events,fragment_rows=[r.tolist() for r in rows],native_uids=[r['observation_uid'] for r in records]))
        if not write_db.execute('SELECT 1 FROM rows WHERE video=? LIMIT 1',(video,)).fetchone():
            write_db.executemany('INSERT INTO rows VALUES (?,?,?,?)',((ordinal,video,uid,payload) for ordinal,uid,payload in source)); write_db.commit()
        count+=len(records); event_count+=len(events)
        write_json(out/'progress.json',dict(stage='native_scores',last_video=video,
            completed_videos=len(reused)+sum(v<=video for v in missing),total_videos=len(shards),
            reused_videos=len(reused),record_count=count,event_count=event_count,pid=os.getpid()))
        print(json.dumps({'stage':'native_scores','index':index+1,'videos':len(shards),'video':video,'events':event_count}),flush=True)
    write_db.execute('CREATE INDEX IF NOT EXISTS by_video ON rows(video)'); write_db.commit(); write_db.close(); read_db.close()
    result=dict(status='COMPLETED',artifact='v9_3_reranker_native_scores',record_count=count,event_count=event_count,video_ids=[int(s['video_id']) for s in shards],
        checkpoint=str(checkpoint),checkpoint_hash=sha256(checkpoint),training_protocol=training['protocol'],NOT_PAPER_VALID=training['NOT_PAPER_VALID'],
        feature_config=config,event_cache_source_metadata=str(Path(cache)/'metadata.json'),source_prediction=meta['frontend_prediction'],source_prediction_hash=meta['frontend_prediction_hash'],
        source_index_original_payloads_only=True,source_index=str(source_index),source_index_hash=sha256(source_index),
        native_manifest=str(subset_manifest or meta['manifest']),native_manifest_hash=sha256(subset_manifest or meta['manifest']),
        gt_used_for_inference=False,gt_used_for_candidate_eligibility=False,decision_frame='last visible query observation',
        source_hashes=source_hashes(),reused_video_ids=reused,newly_scored_video_ids=missing,
        run_binding_hash=sha256(receipt_path),score_file_hashes={p.name:sha256(p) for p in (out/'videos').glob('*.json')})
    write_json(out/'scores.json',result); lock.close(); return result


def calibrate(scores_dir, cache, output):
    """At most four operating points; labels affect thresholds, never weights."""
    root=Path(scores_dir); score_meta=json.loads((root/'scores.json').read_text()); metadata=json.loads((Path(cache)/'metadata.json').read_text())
    if metadata['split']!='test': raise ValueError('this operating-point protocol requires Test Base')
    if metadata['frontend_prediction_hash']!=score_meta['source_prediction_hash']: raise ValueError('calibration source mismatch')
    if sha256(metadata['rows_path'])!=metadata['rows_hash']: raise ValueError('calibration labels hash mismatch')
    winners={}
    for path in sorted((root/'videos').glob('*.json')):
        if sha256(path)!=score_meta['score_file_hashes'][path.name]: raise ValueError('score cache changed')
        doc=json.loads(path.read_text())
        for event in doc['events']:
            order=sorted(range(len(event['scores'])),key=lambda i:(-event['scores'][i],event['candidates'][i])); first=order[0]
            gap=event['scores'][first]-event['scores'][order[1]] if len(order)>1 else float('inf')
            winners[(doc['video_id'],event['target'])]=(event['candidates'][first],event['scores'][first],gap)
    labeled={}; positive_groups=set(); excluded=Counter()
    with Path(metadata['rows_path']).open() as handle:
        for line in handle:
            e=json.loads(line); key=(e['video_id'],e['target_serial'])
            if key not in winners: continue
            if e['target_base'] and e['candidate_base'] and e['label']==1 and e[f'prefilter_rank_b{score_meta["feature_config"]["query_observations"]}']<=64:
                positive_groups.add(key)
            if e['candidate_serial']!=winners[key][0]: continue
            if not (e['target_base'] and e['candidate_base']): excluded['nonbase_winner']+=1; continue
            if e['label']<0: excluded['unknown_winner']+=1; continue
            labeled[key]=(winners[key][1],winners[key][2],int(e['label']))
    candidates=[]
    for margin in (0.0,0.1):
        eligible=sorted((r for r in labeled.values() if r[1]>=margin),key=lambda r:-r[0]); correct=0; accepted=0; points=[]
        for threshold,group in itertools.groupby(eligible,lambda r:r[0]):
            rows=list(group); accepted+=len(rows); correct+=sum(r[2] for r in rows)
            precision=correct/accepted; recall=correct/max(1,len(positive_groups)); f1=2*precision*recall/max(precision+recall,1e-12)
            points.append(dict(threshold=float(threshold),margin_threshold=margin,precision=precision,recall=recall,f1=f1,accepted=accepted,correct=correct))
        if not points: continue
        for reason,pool in [('best_Base_F1',points),('Base_precision90_recall',[p for p in points if p['precision']>=.9])]:
            if not pool: continue
            best=max(pool,key=lambda p:(p['f1'],p['precision'],p['threshold']) if reason=='best_Base_F1' else (p['recall'],p['precision'],p['threshold']))
            if any(c['threshold']==best['threshold'] and c['margin_threshold']==margin for c in candidates): continue
            candidates.append(dict(config_index=len(candidates),reason=reason,**best))
    if not candidates: raise ValueError('no Base-known operating point')
    result=dict(status='COMPLETED',artifact='v9_3_reranker_threshold_candidates',protocol='TEST_BASE_ADAPTED_THRESHOLDS_ONLY',
        checkpoint_hash=score_meta['checkpoint_hash'],scores_hash=sha256(root/'scores.json'),candidates=candidates,
        supervised_winner_groups=len(labeled),positive_Base_groups=len(positive_groups),excluded=dict(excluded),
        novel_used_for_selection=False,test_optimizer_steps=0,margin_choices=[0.,.1],maximum_candidates=4,
        label_policy='known Base target AND known Base candidate; unknown labels never negative',
        gt_used_for_native_candidate_eligibility=False,NOT_PAPER_VALID=score_meta['NOT_PAPER_VALID'])
    write_json(output,result); return result


def materialize_scored(scores_dir, config, output):
    import ijson
    root=Path(scores_dir); out=Path(output); out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((root/'scores.json').read_text()); db=sqlite3.connect(f'file:{root}/source.sqlite?mode=ro',uri=True)
    assigned={}; counts=Counter()
    for path in sorted((root/'videos').glob('*.json')):
        if sha256(path)!=metadata['score_file_hashes'][path.name]: raise ValueError('scored event hash mismatch')
        doc=json.loads(path.read_text()); lookup={uid:json.loads(payload) for uid,payload in db.execute('SELECT uid,payload FROM rows WHERE video=?',(doc['video_id'],))}
        records=[lookup[uid] for uid in doc['native_uids']]
        rows,stats=replay_scores(records,doc['fragment_rows'],doc['events'],config['threshold'],config['margin_threshold']); counts.update(stats)
        assigned.update({r['observation_uid']:r['track_id'] for r in rows})
    before=hashlib.sha256(); after=hashlib.sha256(); count=0
    with (out/'prediction.json').open('x') as handle:
        handle.write('[\n')
        for count,(uid,payload) in enumerate(db.execute('SELECT uid,payload FROM rows ORDER BY ordinal'),1):
            row=json.loads(payload); old={k:v for k,v in row.items() if k!='track_id'}; before.update((json.dumps(old,sort_keys=True)+'\n').encode())
            row['track_id']=assigned.pop(uid); after.update((json.dumps({k:v for k,v in row.items() if k!='track_id'},sort_keys=True)+'\n').encode())
            handle.write((',' if count>1 else '')+json.dumps(row)+'\n')
        handle.write(']\n')
    if assigned or count!=metadata['record_count']: raise AssertionError('row coverage failure')
    with (out/'prediction.json').open('rb') as handle:
        for pair in itertools.zip_longest(db.execute('SELECT payload FROM rows ORDER BY ordinal'),ijson.items(handle,'item',use_float=True)):
            original,row=pair
            if original is None or row is None: raise AssertionError('invariance count mismatch')
            old=json.loads(original[0]); old.pop('track_id'); row.pop('track_id')
            if old!=row: raise AssertionError('non-track field changed')
    db.close()
    receipt=dict(status='COMPLETED',only_track_id_changed=True,record_count=count,statistics=dict(counts),
        original_observation_hash=before.hexdigest(),output_observation_hash=after.hexdigest(),prediction_hash=sha256(out/'prediction.json'),
        checkpoint_hash=metadata['checkpoint_hash'],config=config,gt_used_for_inference=False,NOT_PAPER_VALID=metadata['NOT_PAPER_VALID'])
    write_json(out/'invariance.json',receipt); return out/'prediction.json'


def pilot(scores_dir, cache, pareto_dir, output):
    """Bounded four-point official Base selection; no implicit full-Test launch."""
    from tools.v9_pareto_shortlist import evaluate
    out=Path(output); out.mkdir(parents=True,exist_ok=False); pareto=Path(pareto_dir)
    control_doc=json.loads((pareto/'candidates.json').read_text()); score_meta=json.loads((Path(scores_dir)/'scores.json').read_text())
    if sorted(control_doc['subset_video_ids'])!=sorted(score_meta['video_ids']) or len(score_meta['video_ids'])!=128:
        raise ValueError('pilot/control 128-video subset mismatch')
    if control_doc['input_hashes']['frontend_prediction']!=score_meta['source_prediction_hash']: raise ValueError('pilot/control source mismatch')
    annotation=Path(control_doc['subset_annotation'])
    if sha256(annotation)!=control_doc['subset_annotation_hash']: raise ValueError('subset annotation mismatch')
    candidates=calibrate(scores_dir,cache,out/'candidates.json')['candidates']; rows=[]
    for config in candidates:
        folder=out/f'candidate_{config["config_index"]:02d}'; prediction=materialize_scored(scores_dir,config,folder)
        evaluation=evaluate(annotation,prediction,folder/'evaluation',f'reranker_{config["config_index"]:02d}',1)
        rows.append(dict(config=config,evaluation=str(folder/'evaluation/evaluation.json'),evaluation_hash=sha256(folder/'evaluation/evaluation.json'),
            base=evaluation['parsed']['base'],novel=evaluation['parsed']['novel'],prediction_hash=sha256(prediction)))
        print(json.dumps({'stage':'official_pilot','config_index':config['config_index'],'base_AssocA':evaluation['parsed']['base']['AssocA'],'novel_AssocA':evaluation['parsed']['novel']['AssocA']}),flush=True)
    winner=max(rows,key=lambda r:(r['base']['AssocA'],r['base']['TETA'],-r['config']['config_index']))
    controls=[]; missing=[]
    for candidate in control_doc['candidates']:
        path=pareto/f'candidate_{candidate["config_index"]:02d}'/'evaluation/evaluation.json'
        if not path.exists(): missing.append(candidate['config_index']); continue
        e=json.loads(path.read_text())
        if e.get('status')!='COMPLETED': missing.append(candidate['config_index']); continue
        if e['annotation_hash']!=sha256(annotation): raise ValueError('control annotation mismatch')
        controls.append(dict(config_index=candidate['config_index'],config=candidate['config'],base=e['parsed']['base'],novel=e['parsed']['novel'],
            evaluation=str(path),evaluation_hash=sha256(path),prediction_hash=e['prediction_hash']))
    if not controls: raise ValueError('no completed current PSMR controls')
    current=max(controls,key=lambda r:r['base']['AssocA']); gain=winner['base']['AssocA']-current['base']['AssocA']
    gate='WAITING_FOR_COMPLETE_CONTROLS' if missing else ('FULL_TEST_ELIGIBLE' if gain>0 else 'NO_PILOT_GAIN_STOP')
    result=dict(status='COMPLETED',artifact='v9_3_reranker_pilot_selection',protocol='VAL_BASE_PILOT',NOT_PAPER_VALID=True,
        winner=winner,rows=rows,controls=controls,missing_control_indices=missing,current_PSMR=current,Base_AssocA_gain=gain,
        full_test_gate=gate,full_test_launched=False,selection_metric='128-video Base AssocA, then Base TETA, then lower config_index',
        novel_used_for_selection=False,checkpoint_hash=score_meta['checkpoint_hash'],source_hashes=source_hashes())
    write_json(out/'selection.json',result); return result


def full_test(scores_dir, cache, selection, output):
    """Freeze a completed Base winner, reuse subset scores, then official Test."""
    import shutil
    import atexit
    import datetime
    import threading
    from tools.v9_pareto_shortlist import evaluate
    root=Path(scores_dir); selected=json.loads(Path(selection).read_text())
    scores=json.loads((root/'scores.json').read_text())
    if (selected.get('status')!='COMPLETED' or selected.get('full_test_gate')!='FULL_TEST_ELIGIBLE'
            or selected.get('missing_control_indices') or selected.get('novel_used_for_selection')
            or selected['winner']['base']['AssocA']<=selected['current_PSMR']['base']['AssocA']):
        raise ValueError('full Test requires completed positive Base-only pilot gate')
    if selected['checkpoint_hash']!=scores['checkpoint_hash']:
        raise ValueError('pilot/scoring checkpoint mismatch')
    for row in selected['rows']+selected['controls']:
        if sha256(row['evaluation'])!=row['evaluation_hash']:
            raise ValueError('pilot/control evaluation changed after selection')
    if len(scores['video_ids'])!=128 or scores['status']!='COMPLETED':
        raise ValueError('full Test seed requires completed 128-video scores')
    metadata=json.loads((Path(cache)/'metadata.json').read_text())
    if metadata['split']!='test' or metadata['frontend_prediction_hash']!=scores['source_prediction_hash']:
        raise ValueError('full Test source mismatch')
    memory_guard()
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    phase={'value':'seed_reuse'}; stopped=threading.Event(); heartbeat_lock=threading.Lock()
    def heartbeat(status='RUNNING'):
        progress=out/'native_scores/progress.json'
        with heartbeat_lock:
            write_json(out/'heartbeat.json',dict(status=status,stage=phase['value'],pid=os.getpid(),ppid=os.getppid(),
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                native_progress=json.loads(progress.read_text()) if progress.exists() else None,
                resources=Path('/proc/self/status').read_text(),memory=Path('/proc/meminfo').read_text()))
    def heartbeat_loop():
        while not stopped.wait(30): heartbeat()
    def heartbeat_exit():
        stopped.set(); heartbeat('COMPLETED' if phase['value']=='COMPLETED' else 'EXITED_BEFORE_COMPLETION')
    heartbeat(); threading.Thread(target=heartbeat_loop,daemon=True).start(); atexit.register(heartbeat_exit)
    frozen=dict(status='FROZEN',selection=str(selection),selection_hash=sha256(selection),
        config=selected['winner']['config'],checkpoint_hash=scores['checkpoint_hash'],
        Base_AssocA_gain=selected['Base_AssocA_gain'],novel_used_for_selection=False,
        protocol='VAL_BASE_PILOT',NOT_PAPER_VALID=True,source_hashes=source_hashes())
    write_json(out/'selected_config.json',frozen)
    native=out/'native_scores'; (native/'videos').mkdir(parents=True)
    # Copy original payloads, hard-link immutable completed scores. No inference
    # is repeated on the subset. Resume audits their rows/order before reuse.
    source_db_hash=sha256(root/'source.sqlite')
    shutil.copy2(root/'source.sqlite',native/'source.sqlite')
    if sha256(native/'source.sqlite')!=source_db_hash: raise ValueError('source database copy mismatch')
    for name,digest in scores['score_file_hashes'].items():
        path=root/'videos'/name
        if path.name!=name or sha256(path)!=digest: raise ValueError('seed score hash mismatch')
        os.link(path,native/'videos'/name)
    write_json(out/'seed_reuse.json',dict(subset_scores=str(root),subset_scores_hash=sha256(root/'scores.json'),
        source_sqlite_hash=source_db_hash,score_file_hashes=scores['score_file_hashes'],
        reused_video_ids=scores['video_ids'],completed_videos_rescored=0))
    print(json.dumps(dict(stage='full_Test_frozen',config=frozen['config'],reused_videos=128)),flush=True)
    phase['value']='native_scores'; heartbeat()
    score_dataset(cache,scores['checkpoint'],scores['source_index'],None,native,resume=True,
        legacy_validation=root.parent/'validation.json')
    phase['value']='materialize_prediction'; heartbeat()
    prediction=materialize_scored(native,frozen['config'],out/'prediction')
    phase['value']='official_TETA'; heartbeat()
    evaluation=evaluate(Path(metadata['annotation']),prediction,out/'evaluation','covtrack_reranker_test',1)
    result=dict(status='COMPLETED',artifact='v9_3_reranker_full_Test',protocol='VAL_BASE_PILOT',NOT_PAPER_VALID=True,
        frozen_selection_hash=sha256(out/'selected_config.json'),evaluation=str(out/'evaluation/evaluation.json'),
        evaluation_hash=sha256(out/'evaluation/evaluation.json'),base=evaluation['parsed']['base'],novel=evaluation['parsed']['novel'],
        checkpoint_hash=scores['checkpoint_hash'],prediction_hash=sha256(prediction),novel_used_for_selection=False)
    write_json(out/'result.json',result); phase['value']='COMPLETED'; stopped.set(); heartbeat('COMPLETED'); return result


def build_features(cache, output, *, query_observations=4, top_r=3):
    from .v9_parameter_search import _load_event_cache
    from ..streaming.partial_support import build_anchor_evidence_sequence
    import inspect
    root=Path(cache); out=Path(output)
    if out.exists(): raise FileExistsError(out)
    meta,arrays,_=_load_event_cache(root)
    verified={}
    for key in ('manifest','frontend_prediction','annotation'):
        actual=sha256(meta[key])
        if actual!=meta[key+'_hash']: raise ValueError(f'{key} source hash mismatch')
        verified[key]={'path':meta[key],'sha256':actual}
    if sha256(meta['rows_path'])!=meta['rows_hash']: raise ValueError('event rows hash mismatch')
    if query_observations not in (1,2,4): raise ValueError('schema10 supports B1/B2/B4')
    out.mkdir(parents=True); n=int(meta['events'])
    maps={name:np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
          for name,dtype,shape in [('features','float32',(n,len(FEATURE_NAMES))),('labels','int8',(n,)),
              ('supervision_allowed','bool',(n,)),('event_indices','int64',(n,))]}
    maps['features'][:]=0; maps['labels'][:]=-1; maps['supervision_allowed'][:]=False; maps['event_indices'][:]=-1
    offsets=[0]; videos=[]; written=0; consumed=0; counts=Counter(); seen=set()
    with Path(meta['rows_path']).open() as handle:
        indexed=((i,json.loads(line)) for i,line in enumerate(handle))
        for group_key,group in itertools.groupby(indexed,lambda pair:(pair[1]['video_id'],pair[1]['target_serial'])):
            if group_key in seen: raise ValueError('noncontiguous event group')
            seen.add(group_key); rows=list(group); features=[]; keep=[]
            for index,event in rows:
                consumed+=1
                if int(arrays['label'][index])!=event['label'] or bool(arrays['target_base'][index])!=bool(event['target_base']):
                    raise ValueError('JSONL/array label mismatch')
                rank=int(event[f'prefilter_rank_b{query_observations}'])
                if not 1<=rank<=64: continue
                if event['candidate_last']>=event['target_first'] or not meta['min_gap']<=event['gap']<=meta['max_gap']:
                    raise ValueError('noncausal feature edge')
                m=int(arrays['mem_len'][index]); q=min(query_observations,len(event['query_rows']))
                features.append(candidate_features(arrays['cosine'][index,:q,:m],arrays['evidence'][index,:m],event['gap'],rank,
                    top_r=top_r,max_gap=meta['max_gap']))
                keep.append((index,event))
            if not features: continue
            count=len(features); maps['features'][written:written+count]=add_event_context(features)
            for offset,(index,event) in enumerate(keep):
                y=int(event['label']); allowed=bool(event['target_base'] and event['candidate_base'] and y in (0,1))
                maps['labels'][written+offset]=y; maps['supervision_allowed'][written+offset]=allowed; maps['event_indices'][written+offset]=index
                counts['supervised_positive' if y==1 else 'supervised_negative']+=int(allowed)
                counts['unknown_excluded']+=int(y<0); counts['nonbase_excluded']+=int(not(event['target_base'] and event['candidate_base']))
            written+=count; offsets.append(written); videos.append(group_key[0])
            if len(videos)%1000==0:
                memory_guard(); print(json.dumps({'stage':'features','groups':len(videos),'rows':written}),flush=True)
    if consumed!=n: raise ValueError('cache row coverage mismatch')
    for a in maps.values(): a.flush()
    np.save(out/'offsets.npy',np.asarray(offsets,dtype=np.int64)); np.save(out/'videos.npy',np.asarray(videos,dtype=np.int64))
    protocol='VAL_BASE_PILOT' if meta['split']=='val' else ('BASE_TRAIN' if meta['split'] in ('train','dev') else 'TEST_BASE_THRESHOLD_ONLY')
    schema=dict(feature_names=list(FEATURE_NAMES),evidence_columns=list(EVIDENCE_COLUMNS),
        evidence_source=inspect.getsourcefile(build_anchor_evidence_sequence),evidence_source_hash=sha256(inspect.getsourcefile(build_anchor_evidence_sequence)),
        prohibited_inputs=['label','target_base','candidate_base','gt_identity','Base/Novel_flag'],padding='valid query_rows and mem_len only',
        singleton_context='best_other=self; margin=0; rank_percentile=0')
    write_json(out/'feature_schema.json',schema)
    result=dict(status='COMPLETED',artifact='v9_3_candidate_features',split=meta['split'],frontend=meta['frontend'],protocol=protocol,
        paper_valid=protocol=='BASE_TRAIN',paper_status='NOT_PAPER_VALID' if protocol=='VAL_BASE_PILOT' else protocol,
        base_only_supervision=True,supervision_rule='target_base AND candidate_base AND label in {0,1}',
        feature_names=list(FEATURE_NAMES),feature_config=dict(query_observations=query_observations,top_r=top_r,max_gap=meta['max_gap'],min_gap=meta['min_gap'],candidate_top_k=64,memory_capacity=64),
        row_count=written,allocated_rows=n,groups=len(videos),counts=dict(counts),event_cache=str(root),event_cache_hash=sha256(root/'metadata.json'),
        rows_hash=meta['rows_hash'],source_inputs=verified,feature_schema_hash=sha256(out/'feature_schema.json'),
        array_hashes={p.stem:sha256(p) for p in out.glob('*.npy')})
    write_json(out/'features.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('features'); p.add_argument('--event-cache',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('train'); p.add_argument('--features',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cpu'); p.add_argument('--epochs',type=int,default=12)
    p=sub.add_parser('check'); p.add_argument('--output',required=True)
    p=sub.add_parser('parity'); p.add_argument('--event-cache',required=True); p.add_argument('--features',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('score-native'); p.add_argument('--event-cache',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--source-index',required=True); p.add_argument('--subset-manifest'); p.add_argument('--output',required=True); p.add_argument('--resume',action='store_true'); p.add_argument('--legacy-validation')
    p=sub.add_parser('pilot'); p.add_argument('--scores',required=True); p.add_argument('--event-cache',required=True); p.add_argument('--pareto',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('full-test'); p.add_argument('--scores',required=True); p.add_argument('--event-cache',required=True); p.add_argument('--selection',required=True); p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='features': result=build_features(args.event_cache,args.output)
    elif args.action=='train':
        from ..training.reranker_trainer import train
        result=train(args.features,args.output,device=args.device,epochs=args.epochs)
    elif args.action=='check': result=self_check(args.output)
    elif args.action=='parity': result=native_parity(args.event_cache,args.features,args.checkpoint,args.output)
    elif args.action=='score-native': result=score_dataset(args.event_cache,args.checkpoint,args.source_index,args.subset_manifest,args.output,resume=args.resume,legacy_validation=args.legacy_validation)
    elif args.action=='pilot': result=pilot(args.scores,args.event_cache,args.pareto,args.output)
    elif args.action=='full-test': result=full_test(args.scores,args.event_cache,args.selection,args.output)
    print(json.dumps(result,default=str),flush=True)


if __name__=='__main__': main()
