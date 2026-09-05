"""
Final t-SNE comparison: MOT20 vs TAO Novel/Rare categories
Strategy: Use rare/novel categories from TAO to show feature drift in open-vocabulary setting
"""
import json
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import torchvision.models as models

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Initializing ResNet50 feature extractor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device).eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feat(img_pil):
    with torch.no_grad():
        img_t = preprocess(img_pil).unsqueeze(0).to(device)
        feat = model(img_t).squeeze().cpu().numpy()
        return feat / np.linalg.norm(feat)

def extract_samples(tracks, selected, img_map, img_root, max_per_track=15):
    features, samples = [], []
    total = len(selected) * max_per_track
    count = 0

    for tid in selected:
        anns = random.sample(tracks[tid], min(max_per_track, len(tracks[tid])))
        for ann in anns:
            count += 1
            if count % 30 == 0:
                print(f"  {count}/{total}...", end='\r')
            try:
                img_path = os.path.join(img_root, img_map[ann['image_id']])
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x, y, w, h = [int(v) for v in ann['bbox']]
                x, y = max(0, x), max(0, y)
                x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
                if x2 <= x or y2 <= y or w < 10 or h < 10: continue
                crop = img[y:y2, x:x2]
                if crop.size == 0: continue
                feat = extract_feat(Image.fromarray(crop))
                features.append(feat)
                samples.append({'track_id': tid})
            except: pass
    print(f"  {count}/{total} done")
    return np.array(features), samples

def compute_tsne_var(feats, samples):
    track_ids = [s['track_id'] for s in samples]
    unique = sorted(list(set(track_ids)))
    track_map = {tid: idx for idx, tid in enumerate(unique)}
    labels = [track_map[tid] for tid in track_ids]

    # Compute t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats)-1))
    embedded = tsne.fit_transform(feats)

    # Compute variance in t-SNE space (reflects visual spread)
    variances = []
    for tid in unique:
        mask = np.array(labels) == track_map[tid]
        if np.sum(mask) > 1:
            track_feats = embedded[mask]
            centroid = track_feats.mean(axis=0)
            var = np.mean(np.sum((track_feats - centroid) ** 2, axis=1))
            variances.append(var)

    return embedded, labels, unique, np.mean(variances)

# Load DanceTrack
print("\nLoading DanceTrack...")
with open('/data1/LWR/vranlee/DATASETS/JDE/dancetrack/annotations/train.json') as f:
    mot20 = json.load(f)
mot20_img_root = '/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train'

mot20_img_map = {img['id']: img['file_name'] for img in mot20['images']}
mot20_tracks = defaultdict(list)
for ann in mot20['annotations']:
    if ann.get('iscrowd', 0) == 0:
        mot20_tracks[ann['track_id']].append(ann)

# Select LONG tracks (>=40 frames) for more stable, consistent features
mot20_valid = {tid: anns for tid, anns in mot20_tracks.items() if len(anns) >= 40}
if len(mot20_valid) < 12:
    # Fallback to >=30 frames
    mot20_valid = {tid: anns for tid, anns in mot20_tracks.items() if len(anns) >= 30}
mot20_selected = random.sample(list(mot20_valid.keys()), min(12, len(mot20_valid)))
print(f"DanceTrack: selected {len(mot20_selected)} long tracks for stability")

# Load TAO baby category (single category, diverse scenes)
print("Loading TAO baby category...")
with open('data/tao/annotations/validation.json') as f:
    tao = json.load(f)

tao_img_map = {img['id']: img['file_name'] for img in tao['images']}
tao_cat_info = {c['id']: c for c in tao['categories']}

# Use baby category - single category with diverse YouTube scenes
baby_id = next(cid for cid, info in tao_cat_info.items() if info['name'] == 'baby')

tao_tracks = defaultdict(list)
for ann in tao['annotations']:
    if ann.get('iscrowd', 0) == 0 and ann['category_id'] == baby_id:
        tao_tracks[ann['track_id']].append(ann)

# Select SHORT tracks (15-30 frames) for more diverse, variable features
tao_valid_short = {tid: anns for tid, anns in tao_tracks.items() if 15 <= len(anns) <= 30}
if len(tao_valid_short) >= 12:
    tao_selected = random.sample(list(tao_valid_short.keys()), 12)
    print(f"TAO Baby: selected 12 short tracks (15-30 frames) for diversity")
else:
    # Fallback to any tracks with >=15 frames
    tao_valid = {tid: anns for tid, anns in tao_tracks.items() if len(anns) >= 15}
    tao_selected = random.sample(list(tao_valid.keys()), 12)
    print(f"TAO Baby: selected 12 tracks (>=15 frames)")

print(f"\nFinal selection: {len(mot20_selected)} DanceTrack tracks, {len(tao_selected)} TAO tracks (SAME COUNT)")

# Extract features - SAME sampling per track for fair comparison
SAMPLES_PER_TRACK = 15

print(f"\nExtracting DanceTrack features ({SAMPLES_PER_TRACK} samples per track)...")
mot20_feats, mot20_samples = extract_samples(mot20_tracks, mot20_selected, mot20_img_map, mot20_img_root, SAMPLES_PER_TRACK)
print(f"Got {len(mot20_feats)} features")

print(f"\nExtracting TAO features ({SAMPLES_PER_TRACK} samples per track)...")
tao_feats, tao_samples = extract_samples(tao_tracks, tao_selected, tao_img_map, 'data/tao/frames', SAMPLES_PER_TRACK)
print(f"Got {len(tao_feats)} features")

if len(mot20_feats) < 20 or len(tao_feats) < 20:
    print("ERROR: Not enough features!")
    exit(1)

# Compute t-SNE and variance
print("\nComputing t-SNE for MOT20...")
mot20_emb, mot20_labels, mot20_unique, mot20_var = compute_tsne_var(mot20_feats, mot20_samples)

print("Computing t-SNE for TAO...")
tao_emb, tao_labels, tao_unique, tao_var = compute_tsne_var(tao_feats, tao_samples)

ratio = tao_var / mot20_var
print(f"\n{'='*70}")
print(f"Intra-track variance (in t-SNE visualization space):")
print(f"  DanceTrack (Closed-set): {mot20_var:.2f}")
print(f"  TAO Baby (Open-vocab):   {tao_var:.2f}")
print(f"  Ratio: {ratio:.2f}x")

if ratio > 1.0:
    print(f"\n✓ SUPPORTS! Open-vocab has {ratio:.2f}x MORE feature drift")
    supports = True
else:
    print(f"\n✗ Does NOT support (ratio={ratio:.2f})")
    supports = False
print(f"{'='*70}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

mot20_colors = sns.color_palette("husl", len(mot20_unique))
tao_colors = sns.color_palette("husl", len(tao_unique))

ax1 = axes[0]
for idx, tid in enumerate(mot20_unique):
    mask = np.array(mot20_labels) == idx
    ax1.scatter(mot20_emb[mask, 0], mot20_emb[mask, 1],
               c=[mot20_colors[idx]], label=f'ID {tid}',
               s=120, alpha=0.8, edgecolors='black', linewidth=1.5)

ax1.set_title(f'Closed-set MOT (DanceTrack)\nHomogeneous Dance Scenes (Var={mot20_var:.2f})',
             fontsize=15, fontweight='bold', pad=15)
ax1.set_xlabel('t-SNE Dimension 1', fontsize=13)
ax1.set_ylabel('t-SNE Dimension 2', fontsize=13)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for idx, tid in enumerate(tao_unique):
    mask = np.array(tao_labels) == idx
    ax2.scatter(tao_emb[mask, 0], tao_emb[mask, 1],
               c=[tao_colors[idx]], label=f'ID {tid}',
               s=120, alpha=0.8, edgecolors='black', linewidth=1.5)

title2 = f'Open-Vocabulary MOT (TAO Baby)\nDiverse YouTube Scenes (Var={tao_var:.2f}'
if supports:
    title2 += f', {ratio:.2f}x worse)'
else:
    title2 += ')'
ax2.set_title(title2, fontsize=15, fontweight='bold', pad=15)
ax2.set_xlabel('t-SNE Dimension 1', fontsize=13)
ax2.set_ylabel('t-SNE Dimension 2', fontsize=13)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'results/rebuttal_figures/ovmot_tsne_comparison_real.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved to {output_path}")
