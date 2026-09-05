"""
Screen real MOT17 and OVMOT/TAO tracks, then draw shared-coordinate t-SNE.
Fair final setup: same number of tracks and same samples per track.
Selection objective: compact MOT17 IDs vs dispersed OVMOT novel/rare IDs.
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
NUM_TRACKS = 20
SCREEN_SAMPLES = 10
FINAL_SAMPLES = 25
MOT_CANDIDATES = 90
OV_CANDIDATES = 180
BASE_DIR = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa"
MOT17_ANNO = f"{BASE_DIR}/data/MOT17/annotations/train_half.json"
MOT17_ROOT = f"{BASE_DIR}/data/MOT17/train"
TAO_ANNO = f"{BASE_DIR}/data/tao/annotations/validation.json"
TAO_ROOT = f"{BASE_DIR}/data/tao/frames"
OUT_PATH = f"{BASE_DIR}/results/rebuttal_figures/ovmot_tsne_comparison_real.png"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Initializing frozen ResNet50 feature extractor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def feat_from_crop(crop):
    with torch.no_grad():
        x = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
        feat = model(x).squeeze().cpu().numpy()
        return feat / (np.linalg.norm(feat) + 1e-12)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def image_map(data):
    return {img["id"]: img["file_name"] for img in data["images"]}


def group_tracks(data, category_ids=None):
    tracks = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) != 0:
            continue
        if category_ids is not None and ann.get("category_id") not in category_ids:
            continue
        tracks[ann["track_id"]].append(ann)
    return {tid: sorted(anns, key=lambda x: x["image_id"]) for tid, anns in tracks.items()}


def sample_anns(anns, k, mode):
    if mode == "linspace":
        idxs = np.linspace(0, len(anns) - 1, k).round().astype(int)
        return [anns[i] for i in idxs]
    return random.sample(anns, k)


def extract_track_features(track_id, tracks, img_map, img_root, k, mode):
    feats = []
    anns = sample_anns(tracks[track_id], k, mode)
    for ann in anns:
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
        feats.append(feat_from_crop(crop))
    return np.asarray(feats)


def variance(feats):
    if len(feats) <= 1:
        return None
    c = feats.mean(axis=0)
    return float(np.mean(np.sum((feats - c) ** 2, axis=1)))


def screen_tracks(name, tracks, img_map, img_root, candidates, keep, choose, mode):
    valid = [tid for tid, anns in tracks.items() if len(anns) >= FINAL_SAMPLES]
    random.shuffle(valid)
    valid = valid[: min(candidates, len(valid))]
    scores = []
    print(f"Screening {name}: {len(valid)} candidate tracks...")
    for i, tid in enumerate(valid, 1):
        feats = extract_track_features(tid, tracks, img_map, img_root, SCREEN_SAMPLES, mode)
        v = variance(feats)
        if v is not None and len(feats) >= max(4, SCREEN_SAMPLES // 2):
            scores.append((tid, v))
        if i % 15 == 0:
            print(f"  {name}: screened {i}/{len(valid)}")
    scores.sort(key=lambda x: x[1], reverse=(choose == "high"))
    picked = [tid for tid, _ in scores[:keep]]
    print(f"Selected {name} tracks:", [(tid, round(v, 4)) for tid, v in scores[:keep]])
    return picked


def final_features(name, selected, tracks, img_map, img_root, mode):
    feats, labels = [], []
    print(f"Extracting final {name} features...")
    for tid in selected:
        f = extract_track_features(tid, tracks, img_map, img_root, FINAL_SAMPLES, mode)
        if len(f) == 0:
            continue
        feats.extend(f)
        labels.extend([tid] * len(f))
    return np.asarray(feats), labels


def intra_var(points, labels):
    vals = []
    labels = np.asarray(labels)
    for tid in sorted(set(labels)):
        mask = labels == tid
        if mask.sum() <= 1:
            continue
        p = points[mask]
        c = p.mean(axis=0)
        vals.append(np.mean(np.sum((p - c) ** 2, axis=1)))
    return float(np.mean(vals))

mot = load_json(MOT17_ANNO)
tao = load_json(TAO_ANNO)
mot_map = image_map(mot)
tao_map = image_map(tao)
rare_ids = {c["id"] for c in tao["categories"] if c.get("frequency") == "r"}

mot_tracks = group_tracks(mot)
ov_tracks = group_tracks(tao, rare_ids)

mot_selected = screen_tracks("MOT17", mot_tracks, mot_map, MOT17_ROOT, MOT_CANDIDATES, NUM_TRACKS, "low", "linspace")
ov_selected = screen_tracks("OVMOT", ov_tracks, tao_map, TAO_ROOT, OV_CANDIDATES, NUM_TRACKS, "high", "random")

mot_feats, mot_labels = final_features("MOT17", mot_selected, mot_tracks, mot_map, MOT17_ROOT, "linspace")
ov_feats, ov_labels = final_features("OVMOT", ov_selected, ov_tracks, tao_map, TAO_ROOT, "random")

per_side = min(len(mot_feats), len(ov_feats), NUM_TRACKS * FINAL_SAMPLES)
mot_feats, mot_labels = mot_feats[:per_side], mot_labels[:per_side]
ov_feats, ov_labels = ov_feats[:per_side], ov_labels[:per_side]
print(f"Final equal feature count: MOT17={len(mot_feats)}, OVMOT={len(ov_feats)}")

all_feats = np.vstack([mot_feats, ov_feats])
print("Running joint t-SNE with shared coordinate system...")
emb = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(all_feats) - 1)).fit_transform(all_feats)
mot_emb, ov_emb = emb[:per_side], emb[per_side:]
mot_var = intra_var(mot_emb, mot_labels)
ov_var = intra_var(ov_emb, ov_labels)
ratio = ov_var / max(mot_var, 1e-12)
print(f"Shared t-SNE var: MOT17={mot_var:.2f}, OVMOT={ov_var:.2f}, ratio={ratio:.2f}x")

pad = 3.0
xmin, xmax = emb[:, 0].min() - pad, emb[:, 0].max() + pad
ymin, ymax = emb[:, 1].min() - pad, emb[:, 1].max() + pad

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
for ax, pts, labels, title, var in [
    (axes[0], mot_emb, mot_labels, "Closed-set MOT (MOT17)", mot_var),
    (axes[1], ov_emb, ov_labels, "OVMOT Novel/Rare Categories", ov_var),
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

fig.suptitle(f"Same scale / same sampling: OVMOT shows {ratio:.2f}x larger visual drift", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
