"""
Draw t-SNE comparison between closed-set MOT and open-vocabulary MOT
using real dataset annotations and extracted features.
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
        """Extract feature from PIL image"""
        with torch.no_grad():
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
            features = self.model(image_tensor)
            features = features.squeeze().cpu().numpy()
            features = features / np.linalg.norm(features)
            return features


def load_mot17_data(anno_path, image_root, num_tracks=12, samples_per_track=20):
    """Load MOT17 annotations and sample tracks"""
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
                'category_id': ann.get('category_id', 1),
                'category_name': 'person'
            })

    return samples


def load_dancetrack_data(anno_path, image_root, num_tracks=12, samples_per_track=20):
    """Load DanceTrack annotations and sample tracks"""
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
                'category_id': ann.get('category_id', 1),
                'category_name': 'person'
            })

    return samples


def load_tao_data(anno_path, image_root, num_tracks=12, samples_per_track=20, target_category='person'):
    """Load TAO annotations and sample tracks from person category"""
    with open(anno_path, 'r') as f:
        data = json.load(f)

    image_id_to_path = {}
    for img in data['images']:
        image_id_to_path[img['id']] = img['file_name']

    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    category_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}

    target_cat_id = category_name_to_id.get(target_category)
    if target_cat_id is None:
        print(f"Warning: category '{target_category}' not found, using diverse categories")
        target_cat_id = None

    track_annotations = defaultdict(list)
    track_categories = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 0:
            track_id = ann['track_id']
            cat_id = ann['category_id']
            if target_cat_id is None or cat_id == target_cat_id:
                track_annotations[track_id].append(ann)
                track_categories[track_id] = cat_id

    valid_tracks = {tid: anns for tid, anns in track_annotations.items()
                   if len(anns) >= samples_per_track}

    if len(valid_tracks) < num_tracks:
        print(f"Warning: only {len(valid_tracks)} valid tracks found for category '{target_category}'")
        print("Falling back to diverse categories...")
        return load_tao_data_diverse(anno_path, image_root, num_tracks, samples_per_track)

    selected_tracks = random.sample(list(valid_tracks.keys()),
                                   min(num_tracks, len(valid_tracks)))

    samples = []
    for track_id in selected_tracks:
        anns = valid_tracks[track_id]
        sampled_anns = random.sample(anns, min(samples_per_track, len(anns)))

        for ann in sampled_anns:
            image_path = os.path.join(image_root, image_id_to_path[ann['image_id']])
            bbox = ann['bbox']
            cat_id = ann['category_id']
            samples.append({
                'track_id': track_id,
                'image_path': image_path,
                'bbox': bbox,
                'category_id': cat_id,
                'category_name': category_id_to_name.get(cat_id, 'unknown')
            })

    return samples


def load_tao_data_diverse(anno_path, image_root, num_tracks=12, samples_per_track=20):
    """Load TAO annotations with diverse challenging categories"""
    with open(anno_path, 'r') as f:
        data = json.load(f)

    image_id_to_path = {}
    for img in data['images']:
        image_id_to_path[img['id']] = img['file_name']

    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    track_annotations = defaultdict(list)
    track_categories = {}
    for ann in data['annotations']:
        if ann.get('iscrowd', 0) == 0:
            track_id = ann['track_id']
            track_annotations[track_id].append(ann)
            track_categories[track_id] = ann['category_id']

    valid_tracks = {tid: anns for tid, anns in track_annotations.items()
                   if len(anns) >= samples_per_track}

    category_tracks = defaultdict(list)
    for track_id in valid_tracks.keys():
        cat_id = track_categories[track_id]
        category_tracks[cat_id].append(track_id)

    selected_tracks = []
    categories_sampled = list(category_tracks.keys())
    random.shuffle(categories_sampled)

    for cat_id in categories_sampled:
        if len(selected_tracks) >= num_tracks:
            break
        tracks_in_cat = category_tracks[cat_id]
        selected_tracks.extend(random.sample(tracks_in_cat,
                                            min(1, len(tracks_in_cat))))

    if len(selected_tracks) < num_tracks:
        remaining = list(set(valid_tracks.keys()) - set(selected_tracks))
        selected_tracks.extend(random.sample(remaining,
                                            min(num_tracks - len(selected_tracks),
                                                len(remaining))))

    samples = []
    for track_id in selected_tracks[:num_tracks]:
        anns = valid_tracks[track_id]
        sampled_anns = random.sample(anns, min(samples_per_track, len(anns)))

        for ann in sampled_anns:
            image_path = os.path.join(image_root, image_id_to_path[ann['image_id']])
            bbox = ann['bbox']
            cat_id = ann['category_id']
            samples.append({
                'track_id': track_id,
                'image_path': image_path,
                'bbox': bbox,
                'category_id': cat_id,
                'category_name': category_id_to_name.get(cat_id, 'unknown')
            })

    return samples


def extract_features_from_samples(samples, feature_extractor):
    """Extract features from cropped regions"""
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

            if x2 <= x or y2 <= y:
                continue

            crop = image[y:y2, x:x2]
            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(crop)
            feat = feature_extractor.extract(crop_pil)
            features.append(feat)
            valid_samples.append(sample)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")

        except Exception as e:
            print(f"Error processing {sample['image_path']}: {e}")
            continue

    return np.array(features), valid_samples


def plot_tsne_comparison(mot_features, mot_samples, tao_features, tao_samples, output_path):
    """Plot t-SNE comparison between closed-set MOT and open-vocabulary MOT"""

    mot_track_ids = [s['track_id'] for s in mot_samples]
    unique_mot_tracks = sorted(list(set(mot_track_ids)))
    mot_track_to_idx = {tid: idx for idx, tid in enumerate(unique_mot_tracks)}
    mot_labels = [mot_track_to_idx[tid] for tid in mot_track_ids]

    tao_track_ids = [s['track_id'] for s in tao_samples]
    unique_tao_tracks = sorted(list(set(tao_track_ids)))
    tao_track_to_idx = {tid: idx for idx, tid in enumerate(unique_tao_tracks)}
    tao_labels = [tao_track_to_idx[tid] for tid in tao_track_ids]

    tao_track_to_category = {}
    for sample in tao_samples:
        tao_track_to_category[sample['track_id']] = sample.get('category_name', 'unknown')

    print("Computing t-SNE for MOT17...")
    tsne_mot = TSNE(n_components=2, random_state=42, perplexity=min(30, len(mot_features)-1))
    mot_embedded = tsne_mot.fit_transform(mot_features)

    print("Computing t-SNE for TAO...")
    tsne_tao = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tao_features)-1))
    tao_embedded = tsne_tao.fit_transform(tao_features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    mot_colors = sns.color_palette("husl", len(unique_mot_tracks))
    tao_colors = sns.color_palette("husl", len(unique_tao_tracks))

    ax1 = axes[0]
    for idx, track_id in enumerate(unique_mot_tracks):
        mask = np.array(mot_labels) == idx
        ax1.scatter(mot_embedded[mask, 0], mot_embedded[mask, 1],
                   c=[mot_colors[idx]], label=f'ID {track_id}',
                   s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

    ax1.set_title('Closed-set MOT (MOT17)\nSingle Category: Person',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for idx, track_id in enumerate(unique_tao_tracks):
        mask = np.array(tao_labels) == idx
        category = tao_track_to_category[track_id]
        ax2.scatter(tao_embedded[mask, 0], tao_embedded[mask, 1],
                   c=[tao_colors[idx]], label=f'ID {track_id} ({category})',
                   s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

    ax2.set_title('Open-Vocabulary MOT (TAO)\nMultiple Diverse Categories',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_path}")

    mot_variances = []
    for idx in range(len(unique_mot_tracks)):
        mask = np.array(mot_labels) == idx
        if np.sum(mask) > 1:
            track_features = mot_embedded[mask]
            centroid = track_features.mean(axis=0)
            variance = np.mean(np.sum((track_features - centroid) ** 2, axis=1))
            mot_variances.append(variance)

    tao_variances = []
    for idx in range(len(unique_tao_tracks)):
        mask = np.array(tao_labels) == idx
        if np.sum(mask) > 1:
            track_features = tao_embedded[mask]
            centroid = track_features.mean(axis=0)
            variance = np.mean(np.sum((track_features - centroid) ** 2, axis=1))
            tao_variances.append(variance)

    print(f"\nIntra-track variance (lower is better):")
    print(f"  MOT17 (Closed-set): {np.mean(mot_variances):.2f} ± {np.std(mot_variances):.2f}")
    print(f"  TAO (Open-vocab):   {np.mean(tao_variances):.2f} ± {np.std(tao_variances):.2f}")
    print(f"  Ratio (TAO/MOT17):  {np.mean(tao_variances) / np.mean(mot_variances):.2f}x")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    base_dir = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa"

    mot17_anno = os.path.join(base_dir, "data/MOT17/annotations/train_half.json")
    mot17_images = os.path.join(base_dir, "data/MOT17/train")

    tao_anno = os.path.join(base_dir, "data/tao/annotations/validation.json")
    tao_images = os.path.join(base_dir, "data/tao/frames")

    output_path = os.path.join(base_dir, "results/rebuttal_figures/ovmot_tsne_comparison_real.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Initializing feature extractor...")
    feature_extractor = FeatureExtractor()

    print("\nLoading MOT17 data...")
    mot_samples = load_mot17_data(mot17_anno, mot17_images,
                                  num_tracks=12, samples_per_track=20)
    print(f"Loaded {len(mot_samples)} samples from {len(set(s['track_id'] for s in mot_samples))} tracks")

    print("\nLoading TAO data...")
    tao_samples = load_tao_data(tao_anno, tao_images,
                                num_tracks=12, samples_per_track=20)
    print(f"Loaded {len(tao_samples)} samples from {len(set(s['track_id'] for s in tao_samples))} tracks")
    print(f"Categories: {set(s['category_name'] for s in tao_samples)}")

    print("\nExtracting features from MOT17...")
    mot_features, mot_valid_samples = extract_features_from_samples(mot_samples, feature_extractor)
    print(f"Extracted {len(mot_features)} features")

    print("\nExtracting features from TAO...")
    tao_features, tao_valid_samples = extract_features_from_samples(tao_samples, feature_extractor)
    print(f"Extracted {len(tao_features)} features")

    print("\nPlotting t-SNE comparison...")
    plot_tsne_comparison(mot_features, mot_valid_samples,
                        tao_features, tao_valid_samples,
                        output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
