#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safely apply merge_pairs.json to tao_track.json (per video), with collision handling.

- Step 1: Apply events (video_id, frame, root, child) in ascending (video_id, frame).
          By default, only rename child's items where image_id >= event.frame (cutoff).
- Step 2: Per-frame de-dup:
          For each (video_id, image_id, track_id) with >1 items:
            * if any pair IoU >= iou_thr (default 0.5):
                keep the highest-score item, drop the rest  (they are duplicate for same object)
            * else (all IoU < iou_thr):
                revert the renamed ones in this group back to their original child id
                (so two distinct objects keep separate IDs)

Input formats:
  - merge_pairs.json: {"pairs":[{"video_id":4,"frame":168,"root":7,"child":42,"emd":0.291}, ...]}
  - tao_track.json:   list of detections with keys: image_id, track_id, bbox=[x,y,w,h], score, category_id, video_id

Usage:
  python tools/merge_tracks_with_pairs_safe.py \
    --pred  /path/to/tao_track.json \
    --pairs /path/to/merge_pairs.json \
    --out   /path/to/tao_track_fixed.json \
    --iou-thr 0.5 \
    --no-cutoff   # (optional) if set, rename all frames (ignore event.frame cutoff)
"""

import argparse
import json
import os
from collections import defaultdict

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def normalize_pairs(pairs_raw):
    if isinstance(pairs_raw, dict) and "pairs" in pairs_raw:
        return pairs_raw["pairs"]
    if isinstance(pairs_raw, list):
        return pairs_raw
    raise ValueError("merge_pairs.json must be a list or have key 'pairs'.")

def iou_xywh(a, b):
    # a,b: [x,y,w,h]
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, xi2 - xi1), max(0.0, yi2 - yi1)
    inter = iw * ih
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0

def sort_events(pairs):
    # normalize (root,child) to (min,max) just in case; then sort
    evts = []
    for it in pairs:
        try:
            vid = int(it["video_id"])
            frm = int(it.get("frame", -1))
            r = int(it["root"])
            c = int(it["child"])
        except Exception:
            continue
        if c == r:
            continue
        if c < r:
            r, c = c, r
        evts.append({"video_id": vid, "frame": frm, "root": r, "child": c})
    evts.sort(key=lambda x: (x["video_id"], x["frame"], x["root"], x["child"]))
    return evts

def apply_events_naive(pred, events, use_cutoff=True):
    """
    Apply child->root renaming (in-place) respecting per-event cutoff.
    We also store '_orig_tid' to allow later reversion in collision handling.
    """
    changed = 0
    for obj in pred:
        if isinstance(obj, dict) and "track_id" in obj:
            if "_orig_tid" not in obj:
                obj["_orig_tid"] = int(obj["track_id"])

    # group indices per (vid, child) for efficiency
    idx_by_vid_child = defaultdict(list)
    for i, obj in enumerate(pred):
        if not isinstance(obj, dict):
            continue
        try:
            vid = int(obj["video_id"])
            tid = int(obj["track_id"])
        except Exception:
            continue
        idx_by_vid_child[(vid, tid)].append(i)

    for evt in events:
        vid, frame_cut, root, child = evt["video_id"], evt["frame"], evt["root"], evt["child"]
        idxs = idx_by_vid_child.get((vid, child), [])
        if not idxs:
            continue
        for i in idxs:
            obj = pred[i]
            if not isinstance(obj, dict):
                continue
            # cutoff: only rename samples after the merge frame
            if use_cutoff:
                try:
                    if int(obj["image_id"]) < frame_cut:
                        continue
                except Exception:
                    pass
            obj["track_id"] = int(root)
            changed += 1
        # After first pass, future events may still reference this child; that's fine.

    return changed

def dedup_per_frame(pred, iou_thr=0.5):
    """
    Resolve collisions after renaming. In-place modify pred.
    Returns (num_reverted, num_dropped).

    关键修复：只处理真正的同帧冲突（同一轨迹ID在同一帧出现多次）
    对于跨帧合并的轨迹，不应该被视为冲突！
    """
    # build frame groups: (vid, image_id) -> indices
    frame_groups = defaultdict(list)
    for i, obj in enumerate(pred):
        if not isinstance(obj, dict):
            continue
        try:
            frame_groups[(int(obj["video_id"]), int(obj["image_id"]))].append(i)
        except Exception:
            continue

    reverted, dropped = 0, 0

    for (vid, img), idxs in frame_groups.items():
        # build buckets per track_id in this frame
        by_tid = defaultdict(list)
        for i in idxs:
            try:
                by_tid[int(pred[i]["track_id"])].append(i)
            except Exception:
                continue

        for tid, inds in by_tid.items():
            if len(inds) <= 1:
                continue

            # 检查是否都来自同一个原始轨迹（说明是跨帧合并，不是冲突）
            orig_ids = set()
            for i in inds:
                orig = pred[i].get("_orig_tid", pred[i]["track_id"])
                orig_ids.add(orig)

            # 如果都来自同一个原始ID，说明这不是冲突，是同一轨迹的多次检测
            # 保留最高分的即可，不需要revert
            if len(orig_ids) == 1:
                best_i, best_s = None, float("-inf")
                for i in inds:
                    s = pred[i].get("score", 0.0)
                    try:
                        s = float(s)
                    except Exception:
                        s = 0.0
                    if s > best_s:
                        best_s, best_i = s, i
                for i in inds:
                    if i != best_i:
                        pred[i] = None
                        dropped += 1
                continue

            # compute max IoU among all pairs
            max_iou = 0.0
            n = len(inds)
            for a in range(n):
                for b in range(a + 1, n):
                    i, j = inds[a], inds[b]
                    bb1 = pred[i].get("bbox", [0,0,0,0])
                    bb2 = pred[j].get("bbox", [0,0,0,0])
                    try:
                        max_iou = max(max_iou, iou_xywh(bb1, bb2))
                    except Exception:
                        pass

            if max_iou < iou_thr:
                # 智能回滚：只回滚那些真正来自不同原始轨迹的
                reverted_any = False
                orig_ids = set()
                for i in inds:
                    orig = pred[i].get("_orig_tid", pred[i]["track_id"])
                    orig_ids.add(orig)

                # 如果所有检测框来自同一个原始轨迹，说明是检测偏移，不回滚
                if len(orig_ids) == 1:
                    continue

                # 否则回滚所有非主ID的检测框
                for i in inds:
                    orig = pred[i].get("_orig_tid", pred[i]["track_id"])
                    if orig != tid:
                        pred[i]["track_id"] = int(orig)
                        reverted += 1
                        reverted_any = True
                # 如果没人可回滚（全是同源 id），为了评估合法性保留一条最高分，其他删除
                if not reverted_any:
                    best_i, best_s = None, float("-inf")
                    for i in inds:
                        s = pred[i].get("score", 0.0)
                        try:
                            s = float(s)
                        except Exception:
                            s = 0.0
                        if s > best_s:
                            best_s, best_i = s, i
                    for i in inds:
                        if i != best_i:
                            pred[i] = None
                            dropped += 1
                continue

            # Otherwise: duplicates for same object ⇒ keep highest-score, drop others
            best_i, best_s = None, float("-inf")
            for i in inds:
                s = pred[i].get("score", 0.0)
                try:
                    s = float(s)
                except Exception:
                    s = 0.0
                if s > best_s:
                    best_s, best_i = s, i

            for i in inds:
                if i == best_i:
                    continue
                pred[i] = None
                dropped += 1

    pred[:] = [x for x in pred if x is not None]

    for obj in pred:
        if isinstance(obj, dict) and "_orig_tid" in obj:
            del obj["_orig_tid"]

    return reverted, dropped


def apply_events_safe_second_pass(pred, events, use_cutoff=True):           # 改
    """只在同帧内不存在 root 的情况下，把残留的 child -> root（安全二次兜底）"""   # 改
    from collections import defaultdict                                     # 改
    # 先建每帧已有的 track_id 集合，便于 O(1) 查询                           # 改
    frame_tids = defaultdict(set)                                           # 改
    for obj in pred:                                                        # 改
        if not isinstance(obj, dict):                                       # 改
            continue                                                        # 改
        try:                                                                # 改
            key = (int(obj["video_id"]), int(obj["image_id"]))              # 改
            frame_tids[key].add(int(obj["track_id"]))                       # 改
        except Exception:                                                   # 改
            pass                                                            # 改
    # 建 (vid, child) -> indices 的倒排表                                     # 改
    idx_by_vid_child = defaultdict(list)                                    # 改
    for i, obj in enumerate(pred):                                          # 改
        if not isinstance(obj, dict):                                       # 改
            continue                                                        # 改
        try:                                                                # 改
            vid = int(obj["video_id"])                                      # 改
            tid = int(obj["track_id"])                                      # 改
            idx_by_vid_child[(vid, tid)].append(i)                          # 改
        except Exception:                                                   # 改
            continue                                                        # 改
    changed = 0                                                             # 改
    for evt in events:                                                      # 改
        vid, frame_cut, root, child = evt["video_id"], evt["frame"], evt["root"], evt["child"]  # 改
        idxs = idx_by_vid_child.get((vid, child), [])                       # 改
        if not idxs:                                                        # 改
            continue                                                        # 改
        for i in idxs:                                                      # 改
            obj = pred[i]                                                   # 改
            if not isinstance(obj, dict):                                   # 改
                continue                                                    # 改
            try:                                                            # 改
                img = int(obj["image_id"])                                  # 改
            except Exception:                                               # 改
                continue                                                    # 改
            if use_cutoff and img < frame_cut:                              # 改
                continue                                                    # 改
            key = (vid, img)                                                # 改
            # 仅当该帧内还没有 root，才把 child 改成 root                      # 改
            if root in frame_tids[key]:                                     # 改
                continue                                                    # 改
            # 先记录原始 tid，便于之后冲突回滚                                 # 改
            orig_tid = int(obj.get("track_id", child))                      # 改
            obj["_orig_tid"] = obj.get("_orig_tid", orig_tid)               # 改

            obj["track_id"] = int(root)                                     # 改
            # 维护帧内集合，避免后续重复冲突                                   # 改
            if child in frame_tids[key]:                                    # 改
                try:                                                        # 改
                    frame_tids[key].remove(child)                           # 改
                except KeyError:                                            # 改
                    pass                                                    # 改
            frame_tids[key].add(root)                                       # 改
            changed += 1                                                    # 改

    return changed                                                          # 改


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Path to tao_track.json")
    ap.add_argument("--pairs", required=True, help="Path to merge_pairs.json")
    ap.add_argument("--out", required=False, help="Output json (default: *_fixed.json)")
    ap.add_argument("--iou-thr", type=float, default=0.3, help="IoU threshold for duplicate vs. distinct (降低到0.3更宽松)")
    ap.add_argument("--no-cutoff", action="store_true", help="If set, ignore event.frame cutoff")
    args = ap.parse_args()

    pred = load_json(args.pred)
    pairs = normalize_pairs(load_json(args.pairs))
    events = sort_events(pairs)

    print(f"[INFO] events = {len(events)}")

    # 统计合并前的轨迹数
    unique_tracks_before = len(set(obj['track_id'] for obj in pred if isinstance(obj, dict) and 'track_id' in obj))
    print(f"[INFO] 合并前唯一轨迹数: {unique_tracks_before}")

    changed = apply_events_naive(pred, events, use_cutoff=(not args.no_cutoff))
    print(f"[INFO] renamed (child→root) samples: {changed}")

    reverted, dropped = dedup_per_frame(pred, iou_thr=args.iou_thr)
    print(f"[INFO] reverted (conflict, IoU<{args.iou_thr}): {reverted}")
    print(f"[INFO] dropped duplicates (IoU≥{args.iou_thr}): {dropped}")

    # --- 二次兜底：只在同帧内没有 root 的情况下，再做一轮 child→root ---        # 改
    changed2 = apply_events_safe_second_pass(                               # 改
        pred, events, use_cutoff=(not args.no_cutoff)                       # 改
    )                                                                        # 改
    print(f"[INFO] second-pass safe renamed: {changed2}")                    # 改

    # 二次去重，清理可能新增的同帧重复                                               # 改
    if changed2 > 0:                                                         # 改
        reverted2, dropped2 = dedup_per_frame(pred, iou_thr=args.iou_thr)   # 改
        print(f"[INFO] second-pass reverted: {reverted2}")                  # 改
        print(f"[INFO] second-pass dropped : {dropped2}")                   # 改

    # 统计合并后的轨迹数
    unique_tracks_after = len(set(obj['track_id'] for obj in pred if isinstance(obj, dict) and 'track_id' in obj))
    reduction = unique_tracks_before - unique_tracks_after
    reduction_pct = (reduction / unique_tracks_before * 100) if unique_tracks_before > 0 else 0
    print(f"[INFO] 合并后唯一轨迹数: {unique_tracks_after}")
    print(f"[INFO] 减少的轨迹数: {reduction} ({reduction_pct:.2f}%)")

    # 统计受影响的检测框数量
    total_detections = len([obj for obj in pred if isinstance(obj, dict)])
    affected_detections = changed + changed2 if 'changed2' in locals() else changed
    affected_pct = (affected_detections / total_detections * 100) if total_detections > 0 else 0
    print(f"[INFO] 受影响的检测框: {affected_detections}/{total_detections} ({affected_pct:.2f}%)")

    out = args.out or (os.path.splitext(args.pred)[0] + "_fixed.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(pred, f, ensure_ascii=False)
    print(f"[OK] saved: {out}")

if __name__ == "__main__":
    main()
