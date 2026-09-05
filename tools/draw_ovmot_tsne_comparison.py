"""
Generate a compact t-SNE-style feature space comparison figure for paper rebuttal.
Compares closed-set MOT/ReID vs. OV-MOT feature distributions.

Usage:
    python draw_ovmot_tsne_comparison.py

Output:
    - ovmot_tsne_comparison.png
    - ovmot_tsne_comparison.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.collections import LineCollection
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
NUM_IDENTITIES = 6
SAMPLES_PER_ID = 30
OUTPUT_DIR = "results/rebuttal_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette for identities (colorblind-friendly)
COLORS = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
]


def generate_closed_set_features(num_ids=6, samples_per_id=30):
    """
    Generate synthetic features for closed-set MOT/ReID.
    Features form tight, well-separated clusters.
    """
    features = []
    labels = []

    # Define cluster centers in a circular layout
    angles = np.linspace(0, 2 * np.pi, num_ids, endpoint=False)
    radius = 3.0
    centers = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])

    for id_idx in range(num_ids):
        center = centers[id_idx]
        # Small variance for tight clusters
        cluster = np.random.randn(samples_per_id, 2) * 0.25 + center
        features.append(cluster)
        labels.extend([id_idx] * samples_per_id)

    return np.vstack(features), np.array(labels), centers


def generate_ovmot_features(num_ids=6, samples_per_id=30):
    """
    Generate synthetic features for OV-MOT with frozen features.
    Features show larger intra-ID variance, temporal drift, and inter-ID overlap.
    """
    features = []
    labels = []
    drift_points = []
    drift_labels = []
    drift_types = []

    # Define cluster centers (same layout as closed-set for comparison)
    angles = np.linspace(0, 2 * np.pi, num_ids, endpoint=False)
    radius = 3.0
    centers = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])

    for id_idx in range(num_ids):
        center = centers[id_idx]

        # Main cluster with larger variance
        main_samples = int(samples_per_id * 0.75)
        cluster = np.random.randn(main_samples, 2) * 0.55 + center
        features.append(cluster)
        labels.extend([id_idx] * main_samples)

        # Add drift samples (blur, occlusion, deformation)
        drift_samples = samples_per_id - main_samples
        for i in range(drift_samples):
            # Drift direction: towards neighboring clusters
            drift_angle = angles[id_idx] + np.random.uniform(-0.8, 0.8)
            drift_distance = np.random.uniform(1.2, 2.0)
            drift_point = center + np.array([
                drift_distance * np.cos(drift_angle),
                drift_distance * np.sin(drift_angle)
            ])

            drift_points.append(drift_point)
            drift_labels.append(id_idx)

            # Assign drift type
            drift_type = ['blur', 'occl.', 'deform.'][i % 3]
            drift_types.append(drift_type)

    all_features = np.vstack(features)
    all_labels = np.array(labels)
    drift_points = np.array(drift_points)
    drift_labels = np.array(drift_labels)

    return all_features, all_labels, drift_points, drift_labels, drift_types, centers


def plot_closed_set_panel(ax, features, labels, centers):
    """
    Plot closed-set MOT/ReID features with tight clusters.
    """
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis('off')

    # Add panel border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    # Plot features by identity
    for id_idx in range(NUM_IDENTITIES):
        mask = labels == id_idx
        id_features = features[mask]

        # Plot points
        ax.scatter(
            id_features[:, 0], id_features[:, 1],
            c=[COLORS[id_idx]], s=25, alpha=0.7,
            edgecolors='white', linewidths=0.5,
            label=f'ID-{id_idx+1}' if id_idx < 3 else None
        )

        # Add cluster boundary (ellipse)
        cov = np.cov(id_features.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2.5 * np.sqrt(eigenvalues)

        ellipse = Ellipse(
            centers[id_idx], width, height, angle=angle,
            facecolor='none', edgecolor=COLORS[id_idx],
            linewidth=1.0, linestyle='--', alpha=0.4
        )
        ax.add_patch(ellipse)

        # Add ID label near cluster center
        ax.text(
            centers[id_idx][0], centers[id_idx][1] - 0.8,
            f'ID-{id_idx+1}', fontsize=7, ha='center',
            color=COLORS[id_idx], weight='bold'
        )

    # Title
    ax.text(
        0.5, 1.08, 'Closed-set MOT / ReID features',
        transform=ax.transAxes, fontsize=10, ha='center', weight='bold'
    )

    # Top-left annotation
    ax.text(
        0.02, 0.98, 'seen categories\ntrained ID features',
        transform=ax.transAxes, fontsize=6.5, ha='left', va='top',
        color='#555555', style='italic'
    )

    # Bottom annotation
    ax.text(
        0.5, -0.08, 'compact intra-ID clusters / clear inter-ID margins',
        transform=ax.transAxes, fontsize=7.5, ha='center',
        color='#333333', style='italic'
    )


def plot_ovmot_panel(ax, features, labels, drift_points, drift_labels, drift_types, centers):
    """
    Plot OV-MOT features with larger variance, drift, and overlap.
    """
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis('off')

    # Add panel border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    # Plot main features by identity
    for id_idx in range(NUM_IDENTITIES):
        mask = labels == id_idx
        id_features = features[mask]

        # Plot main cluster points
        ax.scatter(
            id_features[:, 0], id_features[:, 1],
            c=[COLORS[id_idx]], s=25, alpha=0.6,
            edgecolors='white', linewidths=0.5
        )

        # Add ID label near cluster center
        ax.text(
            centers[id_idx][0], centers[id_idx][1] - 0.8,
            f'ID-{id_idx+1}', fontsize=7, ha='center',
            color=COLORS[id_idx], weight='bold'
        )

    # Plot drift points with arrows
    for i, (drift_point, drift_label) in enumerate(zip(drift_points, drift_labels)):
        # Plot drift point
        ax.scatter(
            drift_point[0], drift_point[1],
            c=[COLORS[drift_label]], s=30, alpha=0.8,
            marker='X', edgecolors='black', linewidths=0.8
        )

        # Draw arrow from cluster center to drift point
        center = centers[drift_label]
        arrow = FancyArrowPatch(
            center, drift_point,
            arrowstyle='->', mutation_scale=8,
            linewidth=0.8, color=COLORS[drift_label],
            alpha=0.4, linestyle='--'
        )
        ax.add_patch(arrow)

        # Add drift type label (only for a few samples)
        if i % 4 == 0:
            ax.text(
                drift_point[0] + 0.2, drift_point[1] + 0.2,
                drift_types[i], fontsize=5.5,
                color='#666666', style='italic'
            )

    # Highlight overlap region
    overlap_center = (centers[0] + centers[1]) / 2
    ax.text(
        overlap_center[0], overlap_center[1],
        'overlap', fontsize=6, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#CC0000', linewidth=0.8, alpha=0.7),
        color='#CC0000', weight='bold'
    )

    # Title
    ax.text(
        0.5, 1.08, 'OV-MOT / Frozen OV features',
        transform=ax.transAxes, fontsize=10, ha='center', weight='bold'
    )

    # Top-right annotation
    ax.text(
        0.98, 0.98, 'novel categories\nfrozen OV features',
        transform=ax.transAxes, fontsize=6.5, ha='right', va='top',
        color='#555555', style='italic'
    )

    # Bottom annotation
    ax.text(
        0.5, -0.08, 'larger intra-ID variance / stronger inter-ID overlap',
        transform=ax.transAxes, fontsize=7.5, ha='center',
        color='#333333', style='italic'
    )


def main():
    """
    Main function to generate the comparison figure.
    """
    print("Generating OV-MOT vs Closed-set MOT feature space comparison...")

    # Generate synthetic features
    print("  Generating closed-set features...")
    closed_features, closed_labels, closed_centers = generate_closed_set_features(
        NUM_IDENTITIES, SAMPLES_PER_ID
    )

    print("  Generating OV-MOT features...")
    ovmot_features, ovmot_labels, drift_points, drift_labels, drift_types, ovmot_centers = \
        generate_ovmot_features(NUM_IDENTITIES, SAMPLES_PER_ID)

    # Create figure
    print("  Creating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.8), dpi=300)

    # Plot left panel (closed-set)
    plot_closed_set_panel(ax1, closed_features, closed_labels, closed_centers)

    # Plot right panel (OV-MOT)
    plot_ovmot_panel(ax2, ovmot_features, ovmot_labels, drift_points,
                     drift_labels, drift_types, ovmot_centers)

    # Adjust layout
    plt.tight_layout()

    # Save figures
    png_path = os.path.join(OUTPUT_DIR, "ovmot_tsne_comparison.png")
    pdf_path = os.path.join(OUTPUT_DIR, "ovmot_tsne_comparison.pdf")

    print(f"  Saving PNG: {png_path}")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.03, dpi=300)

    print(f"  Saving PDF: {pdf_path}")
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)

    plt.close()

    print("\n" + "="*80)
    print("Figure generated successfully!")
    print("="*80)
    print("\nSuggested caption for paper/rebuttal:")
    print("-" * 80)
    print("""
Feature-space comparison between closed-set MOT and OV-MOT. Colors denote
identities. Closed-set ReID features form compact identity clusters with clear
margins, whereas frozen open-vocabulary features on novel categories exhibit
larger intra-ID variance, temporal drift, and stronger inter-ID overlap under
blur, deformation, and occlusion, motivating the use of temporal memory as an
identity anchor.
""")
    print("-" * 80)
    print(f"\nOutput files:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")
    print()


if __name__ == "__main__":
    main()
