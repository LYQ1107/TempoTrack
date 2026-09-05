"""
Draw t-SNE comparison between closed-set MOT and open-vocabulary MOT
Strategy: Use same object type (person) but different scenarios to show feature drift
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

try:
    import torchvision.models as models
except ImportError:
    print("torchvision not available")
    exit(1)


class FeatureExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_pil):
        with torch.no_grad():
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
            features = self.model(image_tensor)
            features = features.squeeze().cpu().numpy()
            features = features / np.linalg.norm(features)
            return features


def load_dancetrack_data(anno_path, image_root, num_tracks=15, samples_per_track=15):
    """Load DanceTrack - closed-set MOT with single scenario"""
    with open(anno_path, 'r') as f:
        data = json.load(f)

    image_id_to_path = {}
    for img in data['images']:
        image_id_to_path[img['id']] = img['file_name']

    track_annotations = defaultdict(list)
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 0:
            track_annotations[ann['track_id']].append(ann)

    valid_tracks = {tid: anns for tid, anns in track_annotations.items()
                   if len(anns) >= samples_per_track}

    selected_tracks = random.sample(list(valid_tracks.keys()),
                                   min(num_tracks, len(valid_tracks)))

    samples = []
    for track_id in selected_tracks:
        anns = valid_tracks[track_id]
        sampled_anns = random.sample(anns, min(samples_per_track, len(anns)))

        for ann in sampled_anns:
            image_path = os.path.join(image_root, image_id_to_path[ann['image_id']])
            bbox = ann['bbox']
            samples.append({
                'track_id': track_id,
                'image_path': image_path,
                'bbox': bbox,
                'category_name': 'person'
            })

    return samples


def load_tao_person_data(anno_path, image_root, num_tracks=15, samples_per_track=15):
    """Load TAO person tracks - open-vocabulary with diverse scenarios"""
    with open(anno_path, 'r') as f:
        data = json.load(f)

    image_id_to_path = {}
    for img in data['images']:
        image_id_to_path[img['id']] = img['file_name']

    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    category_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}

    person_cat_id = category_name_to_id.get('person')
    if person_cat_id is None:
        print("Warning: 'person' category not found in TAO")
        return []

    track_annotations = defaultdict(list)
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 0 and ann['category_id'] == person_cat_id:
            track_annotations[ann['track_id']].append(ann)

    valid_tracks = {tid: anns for tid, anns in track_annotations.items()
                   if len(anns) >= samples_per_track}

    print(f"Found {len(valid_tracks)} valid person tracks in TAO")

    if len(valid_tracks) < num_tracks:
        print(f"Not enough person tracks, using all {len(valid_tracks)} available")
        selected_tracks = list(valid_tracks.keys())
    else:
        selected_tracks = random.sample(list(valid_tracks.keys()), num_tracks)

    samples = []
    for track_id in selected_tracks:
        anns = valid_tracks[track_id]
        sampled_anns = random.sample(anns, min(samples_per_track, len(anns)))

        for ann in sampled_anns:
            image_path = os.path.join(image_root, image_id_to_path[ann['image_id']])
            bbox = ann['bbox']
            samples.append({
                'track_id': track_id,
                'image_path': image_path,
                'bbox': bbox,
                'category_name': 'person'
            })

    return samples


def extract_features_from_samples(samples, feature_extractor):
    features = []
    valid_samples = []

    for i, sample in enumerate(samples):
        try:
            image = cv2.imread(sample['image_path'])
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            x, y, w, h = [int(v) for v in sample['bbox']]
            x = max(0, x)
            y = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)

            if x2 <= x or y2 <= y or w < 10 or h < 10:
                continue

            crop = image[y:y2, x:x2]
            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(crop)
            feat = feature_extractor.extract(crop_pil)
            features.append(feat)
            valid_samples.append(sample)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")

        except Exception as e:
            continue

    return np.array(features), valid_samples


def compute_intra_track_variance(embedded, labels):
    """Compute average intra-track variance"""
    unique_labels = sorted(list(set(labels)))
    variances = []

    for label in unique_labels:
        mask = np.array(labels) == label
        if np.sum(mask) > 1:
            track_features = embedded[mask]
            centroid = track_features.mean(axis=0)
            variance = np.mean(np.sum((track_features - centroid) ** 2, axis=1))
            variances.append(variance)

    return variances


def plot_tsne_comparison(closed_features, closed_samples, open_features, open_samples,
                        output_path, closed_name="Closed-set MOT", open_name="Open-Vocabulary MOT"):

    closed_track_ids = [s['track_id'] for s in closed_samples]
    unique_closed_tracks = sorted(list(set(closed_track_ids)))
    closed_track_to_idx = {tid: idx for idx, tid in enumerate(unique_closed_tracks)}
    closed_labels = [closed_track_to_idx[tid] for tid in closed_track_ids]

    open_track_ids = [s['track_id'] for s in open_samples]
    unique_open_tracks = sorted(list(set(open_track_ids)))
    open_track_to_idx = {tid: idx for idx, tid in enumerate(unique_open_tracks)}
    open_labels = [open_track_to_idx[tid] for tid in open_track_ids]

    print(f"Computing t-SNE for {closed_name}...")
    tsne_closed = TSNE(n_components=2, random_state=42, perplexity=min(30, len(closed_features)-1))
    closed_embedded = tsne_closed.fit_transform(closed_features)

    print(f"Computing t-SNE for {open_name}...")
    tsne_open = TSNE(n_components=2, random_state=42, perplexity=min(30, len(open_features)-1))
    open_embedded = tsne_open.fit_transform(open_features)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    closed_colors = sns.color_palette("husl", len(unique_closed_tracks))
    open_colors = sns.color_palette("husl", len(unique_open_tracks))

    ax1 = axes[0]
    for idx, track_id in enumerate(unique_closed_tracks):
        mask = np.array(closed_labels) == idx
        ax1.scatter(closed_embedded[mask, 0], closed_embedded[mask, 1],
                   c=[closed_colors[idx]], label=f'ID {track_id}',
                   s=120, alpha=0.8, edgecolors='black', linewidth=1.5)

    ax1.set_title(f'{closed_name}\nSingle Scenario, Consistent Features',
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for idx, track_id in enumerate(unique_open_tracks):
        mask = np.array(open_labels) == idx
        ax2.scatter(open_embedded[mask, 0], open_embedded[mask, 1],
                   c=[open_colors[idx]], label=f'ID {track_id}',
                   s=120, alpha=0.8, edgecolors='black', linewidth=1.5)

    ax2.set_title(f'{open_name}\nDiverse Scenarios, Feature Drift',
                 fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")

    closed_variances = compute_intra_track_variance(closed_embedded, closed_labels)
    open_variances = compute_intra_track_variance(open_embedded, open_labels)

    print(f"\nIntra-track variance (lower = more compact):")
    print(f"  {closed_name}: {np.mean(closed_variances):.2f} ± {np.std(closed_variances):.2f}")
    print(f"  {open_name}:   {np.mean(open_variances):.2f} ± {np.std(open_variances):.2f}")

    if np.mean(open_variances) > np.mean(closed_variances):
        ratio = np.mean(open_variances) / np.mean(closed_variances)
        print(f"  Ratio (Open/Closed): {ratio:.2f}x MORE feature drift in open-vocabulary setting ✓")
        return True
    else:
        ratio = np.mean(closed_variances) / np.mean(open_variances)
        print(f"  Ratio (Closed/Open): {ratio:.2f}x - This comparison doesn't support the argument")
        return False


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    base_dir = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa"

    dancetrack_anno = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack/annotations/train.json"
    dancetrack_images = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train"

    tao_anno = os.path.join(base_dir, "data/tao/annotations/validation.json")
    tao_images = os.path.join(base_dir, "data/tao/frames")

    output_path = os.path.join(base_dir, "results/rebuttal_figures/ovmot_tsne_comparison_real.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Initializing feature extractor...")
    feature_extractor = FeatureExtractor()

    print("\n" + "="*80)
    print("Strategy: DanceTrack (closed-set) vs TAO Person (open-vocabulary)")
    print("="*80)

    print("\nLoading DanceTrack data (closed-set MOT)...")
    dance_samples = load_dancetrack_data(dancetrack_anno, dancetrack_images,
                                        num_tracks=15, samples_per_track=15)
    print(f"Loaded {len(dance_samples)} samples from {len(set(s['track_id'] for s in dance_samples))} tracks")

    print("\nLoading TAO person data (open-vocabulary MOT)...")
    tao_samples = load_tao_person_data(tao_anno, tao_images,
                                      num_tracks=15, samples_per_track=15)
    print(f"Loaded {len(tao_samples)} samples from {len(set(s['track_id'] for s in tao_samples))} tracks")

    if len(tao_samples) == 0:
        print("ERROR: No TAO person samples found!")
        return

    print("\nExtracting features from DanceTrack...")
    dance_features, dance_valid = extract_features_from_samples(dance_samples, feature_extractor)
    print(f"Extracted {len(dance_features)} features")

    print("\nExtracting features from TAO...")
    tao_features, tao_valid = extract_features_from_samples(tao_samples, feature_extractor)
    print(f"Extracted {len(tao_features)} features")

    if len(dance_features) < 30 or len(tao_features) < 30:
        print("ERROR: Not enough features extracted!")
        return

    print("\nPlotting t-SNE comparison...")
    success = plot_tsne_comparison(dance_features, dance_valid,
                                   tao_features, tao_valid,
                                   output_path,
                                   closed_name="DanceTrack (Closed-set)",
                                   open_name="TAO Person (Open-Vocabulary)")

    if success:
        print("\n✓ This comparison SUPPORTS your argument!")
    else:
        print("\n✗ This comparison does NOT support your argument well.")

    print("\nDone!")


if __name__ == "__main__":
    main()
