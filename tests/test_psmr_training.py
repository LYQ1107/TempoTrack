import json
import numpy as np
import torch

from tempotrack_research.streaming.psmr_dataset import VideoData, build_base_episodes
from tempotrack_research.training.psmr_trainer import train_psmr


def test_real_rank_reliability_training_writes_checkpoint(tmp_path):
    n = 8
    features = np.asarray([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]], np.float32)
    video = VideoData(1, features, np.tile(np.asarray([[0, 0, 10, 10]], np.float32), (n, 1)), np.ones(n, np.float32), np.arange(n), np.ones(n, np.int64), np.asarray([0, 0, 1, 1, 2, 2, 3, 3]), np.ones(n, bool), np.asarray([1, 1, 2, 2, 1, 1, 2, 2]), np.ones(n, bool), np.zeros(n, bool), [str(i) for i in range(n)])
    episodes = tmp_path / "episodes.jsonl"
    built = build_base_episodes([video], episodes, max_gap=60)
    assert built["episodes"] > 0
    result = train_psmr(episodes_path=episodes, videos={1: video}, run_root=tmp_path / "run", config={"partial_support": {"top_r": 1, "memory_capacity": 64}, "training": {"save_every": 2, "log_every": 1}}, max_steps=2, device="cpu", resume="never", input_hash="test")
    assert result["status"] == "COMPLETED"
    assert result["optimizer_steps"] == 2
    assert json.loads((tmp_path / "run/train_result.json").read_text())["algorithm_revision"] == "per_anchor_v8"
    assert (tmp_path / "run/last.pt").exists()
