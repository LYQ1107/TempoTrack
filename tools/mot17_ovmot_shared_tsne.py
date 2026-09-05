"""
MOT17 vs OVMOT feature distribution with shared t-SNE coordinates.
- Equal number of tracks and samples per track.
- Joint t-SNE on combined features.
- Shared x/y axis limits for both panels.
"""
import json
import os
import random
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.manifold import TSNE

SEED = 42
NUM_TRACKS = 12
SAMPLES_PER_TRACK = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa"
MOT17_ANNO = f"{BASE_DIR}/data/MOT17/annotations/train_half.json"
MOT17_ROOT = f"{BASE_DIR}/data/MOT17/train"
TAO_ANNO = f"{BASE_DIR}/data/tao/annotations/validation.json"
TAO_ROOT = f"{BASE_DIR}/data/tao/frames"
OUT_PATH = f"{BASE_DIR}/results/rebuttal_figures/ovmot_tsne_comparison_real.png"

print("Initializing frozen ResNet50 feature extractor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_feat(crop_rgb):
    with torch.no_grad():
        x = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
        feat = model(x).squeeze().cpu().numpy()
        return feat / (np.linalg.norm(feat) + 1e-12)


def group_tracks(data, category_filter=None):
    tracks = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) != 0:
            continue
        if category_filter is not None and ann.get("category_id") not in category_filter:
            continue
        tracks[ann["track_id"]].append(ann)
    return tracks


def sample_tracks(tracks, num_tracks, samples_per_track, mode="random"):
    valid = {tid: anns for tid, anns in tracks.items() if len(anns) >= samples_per_track}
    if len(valid) < num_tracks:
        raise RuntimeError(f"Only {len(valid)} valid tracks, need {num_tracks}")
    tids = list(valid.keys())
    if mode == "longest":
        tids = sorted(tids, key=lambda t: len(valid[t]), reverse=True)
        selected = tids[:num_tracks]
    else:
        selected = random.sample(tids, num_tracks)
    samples = []
    for tid in selected:
        anns = sorted(valid[tid], key=lambda a: a["image_id"])
        if mode == "longest":
            idxs = np.linspace(0, len(anns) - 1, samples_per_track).round().astype(int)
            picked = [anns[i] for i in idxs]
        else:
            picked = random.sample(anns, samples_per_track)
        for ann in picked:
            samples.append({"track_id": tid, "ann": ann})
    return samples


def extract_samples(samples, img_map, img_root, label):
    feats, kept = [], []
    for i, sample in enumerate(samples, 1):
        ann = sample["ann"]
        img_path = os.path.join(img_root, img_map[ann["image_id"]])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = [int(v) for v in ann["bbox"]]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x2 <= x or y2 <= y or w < 10 or h < 10:
            continue
        crop = img[y:y2, x:x2]
        if crop.size == 0:
            continue
        feats.append(extract_feat(crop))
        kept.append({"track_id": sample["track_id"], "dataset": label})
        if i % 30 == 0:
            print(f"  {label}: {i}/{len(samples)}")
    return np.asarray(feats), kept


def intra_var(points, labels):
    vals = []
    for lab in sorted(set(labels)):
        mask = np.asarray(labels) == lab
        if mask.sum() <= 1:
            continue
        p = points[mask]
        c = p.mean(axis=0)
        vals.append(np.mean(np.sum((p - c) ** 2, axis=1)))
    return float(np.mean(vals))

print("Loading MOT17...")
with open(MOT17_ANNO) as f:
    mot = json.load(f)
mot_img_map = {img["id"]: img["file_name"] for img in mot["images"]}
mot_tracks = group_tracks(mot)
mot_samples = sample_tracks(mot_tracks, NUM_TRACKS, SAMPLES_PER_TRACK, mode="longest")

print("Loading OVMOT/TAO novel categories...")
with open(TAO_ANNO) as f:
    tao = json.load(f)
tao_img_map = {img["id"]: img["file_name"] for img in tao["images"]}
rare_ids = {c["id"] for c in tao["categories"] if c.get("frequency") == "r"}
tao_tracks = group_tracks(tao, category_filter=rare_ids)
tao_samples = sample_tracks(tao_tracks, NUM_TRACKS, SAMPLES_PER_TRACK, mode="random")

print(f"Equal setup: {NUM_TRACKS} tracks x {SAMPLES_PER_TRACK} samples for each dataset")
print("Extracting MOT17 features...")
mot_feats, mot_kept = extract_samples(mot_samples, mot_img_map, MOT17_ROOT, "MOT17")
print("Extracting OVMOT/TAO features...")
tao_feats, tao_kept = extract_samples(tao_samples, tao_img_map, TAO_ROOT, "OVMOT")

n = min(len(mot_feats), len(tao_feats))
mot_feats, mot_kept = mot_feats[:n], mot_kept[:n]
tao_feats, tao_kept = tao_feats[:n], tao_kept[:n]
print(f"Final equal feature count: {n} MOT17, {n} OVMOT")

all_feats = np.vstack([mot_feats, tao_feats])
print("Running joint t-SNE on combined features...")
emb = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(all_feats) - 1)).fit_transform(all_feats)
mot_emb, tao_emb = emb[:n], emb[n:]

mot_labels = [x["track_id"] for x in mot_kept]
tao_labels = [x["track_id"] for x in tao_kept]
mot_var = intra_var(mot_emb, mot_labels)
tao_var = intra_var(tao_emb, tao_labels)
ratio = tao_var / max(mot_var, 1e-12)
print(f"MOT17 var={mot_var:.2f}, OVMOT var={tao_var:.2f}, ratio={ratio:.2f}x")

pad = 3.0
xmin, xmax = emb[:, 0].min() - pad, emb[:, 0].max() + pad
ymin, ymax = emb[:, 1].min() - pad, emb[:, 1].max() + pad

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
for ax, pts, labels, title, var in [
    (axes[0], mot_emb, mot_labels, "Closed-set MOT (MOT17)", mot_var),
    (axes[1], tao_emb, tao_labels, "OVMOT Novel/Rare Categories", tao_var),
]:
    unique = sorted(set(labels))
    colors = sns.color_palette("husl", len(unique))
    for i, tid in enumerate(unique):
        mask = np.asarray(labels) == tid
        ax.scatter(pts[mask, 0], pts[mask, 1], s=95, alpha=0.82,
                   c=[colors[i]], edgecolors="black", linewidth=1.0,
                   label=f"ID {tid}")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nVar={var:.2f}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Joint t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("Joint t-SNE Dimension 2", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

fig.suptitle(f"Frozen ResNet50 Features: OVMOT shows {ratio:.2f}x larger visual drift", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
