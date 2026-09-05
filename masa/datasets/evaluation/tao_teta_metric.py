import os
import os.path as osp
import pickle
import shutil
import tempfile
from collections import defaultdict
from itertools import chain
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import tqdm

try:
    import teta
except ImportError:
    teta = None

import mmengine
import mmengine.fileio as fileio
from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation.metrics.base_video_metric import BaseVideoMetric
from mmdet.registry import METRICS, TASK_UTILS
from mmengine.dist import (
    all_gather_object, barrier, broadcast, broadcast_object_list, get_dist_info,
    is_main_process)
from mmengine.logging import MMLogger


def get_tmpdir() -> str:
    """return the same tmpdir for all processes."""
    rank, world_size = get_dist_info()
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8)
    if rank == 0:
        tmpdir = tempfile.mkdtemp()
        tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8)
        dir_tensor[: len(tmpdir)] = tmpdir
    broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    return tmpdir


@METRICS.register_module()
class TaoTETAMetric(BaseVideoMetric):
    TRACKER = "masa-tracker"
    allowed_metrics = ["TETA"]
    default_prefix: Optional[str] = "tao_teta_metric"

    def __init__(
        self,
        metric: Union[str, List[str]] = ["TETA"],
        outfile_prefix: Optional[str] = None,
        track_iou_thr: float = 0.5,
        format_only: bool = False,
        ann_file: Optional[str] = None,
        dataset_type: str = "Taov1Dataset",
        use_postprocess: bool = False,
        postprocess_tracklet_cfg: Optional[List[dict]] = [],
        collect_device: str = "cpu",
        tcc: bool = True,
        open_vocabulary=False,
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if teta is None:
            raise RuntimeError(
                "teta is not installed, please install it by: "
                "python -m pip install git+https://github.com/SysCV/tet.git/#subdirectory=teta "
            )

        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError("metric must be a list or a str.")
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")
        self.metrics = metrics
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, (
                'outfile_prefix must be not None when format_only is True, '
                'otherwise the result files will be saved to a temp directory '
                'which will be cleaned up at the end.')
        self.use_postprocess = use_postprocess
        self.postprocess_tracklet_cfg = postprocess_tracklet_cfg.copy()
        self.postprocess_tracklet_methods = [
            TASK_UTILS.build(cfg) for cfg in self.postprocess_tracklet_cfg
        ]
        self.track_iou_thr = track_iou_thr
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir.name = get_tmpdir()
        self.seq_pred = defaultdict(lambda: [])
        self.outfile_prefix = outfile_prefix

        self.ann_file = ann_file
        self.dataset_type = dataset_type
        self.tcc = tcc
        self.open_vocabulary = open_vocabulary

        # --- Lazy-loaded attributes ---
        self.coco = None
        self.cat_ids = None
        self.class_list = None

    def _lazy_init(self):
        """Lazy initialization for attributes that require loading large files."""
        if self.coco is None:
            logger: MMLogger = MMLogger.get_current_instance()
            logger.info('Lazily initializing COCO API for TETA metric...')
            with fileio.get_local_path(self.ann_file) as local_path:
                self.coco = COCO(local_path)

            assert self.dataset_type in ["Taov05Dataset", "Taov1Dataset"]
            if self.dataset_type == "Taov05Dataset":
                from masa.datasets import Taov05Dataset
                self.class_list = Taov05Dataset.METAINFO["classes"]
            if self.dataset_type == "Taov1Dataset":
                from masa.datasets import Taov1Dataset
                self.class_list = Taov1Dataset.METAINFO["classes"]
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.class_list)

    def __del__(self):
        self.tmp_dir.cleanup()

    def process_video(self, data_samples):
        self._lazy_init()
        video_len = len(data_samples)
        for frame_id in range(video_len):
            img_data_sample = data_samples[frame_id].to_dict()
            video_id = img_data_sample["video_id"]
            pred_instances = img_data_sample["pred_track_instances"]
            pred_instances_list = []
            for i in range(len(pred_instances["instances_id"])):
                data_dict = dict()
                data_dict["image_id"] = img_data_sample["img_id"]
                data_dict["track_id"] = int(pred_instances["instances_id"][i])
                data_dict["bbox"] = self.xyxy2xywh(pred_instances["bboxes"][i])
                data_dict["score"] = float(pred_instances["scores"][i])
                data_dict["category_id"] = self.cat_ids[pred_instances["labels"][i]]
                data_dict["video_id"] = img_data_sample["video_id"]
                pred_instances_list.append(data_dict)
            self.seq_pred[video_id].extend(pred_instances_list)

    def process_image(self, data_samples, video_len):
        """Process a single frame when sampling is image-based."""
        self._lazy_init()
        img_data_sample = data_samples[0].to_dict()
        video_id = img_data_sample["video_id"]
        pred_instances = img_data_sample["pred_track_instances"]
        pred_instances_list = []
        for i in range(len(pred_instances["instances_id"])):
            data_dict = dict()
            data_dict["image_id"] = img_data_sample["img_id"]
            data_dict["track_id"] = int(pred_instances["instances_id"][i])
            data_dict["bbox"] = self.xyxy2xywh(pred_instances["bboxes"][i])
            data_dict["score"] = float(pred_instances["scores"][i])
            data_dict["category_id"] = self.cat_ids[pred_instances["labels"][i]]
            data_dict["video_id"] = img_data_sample["video_id"]
            pred_instances_list.append(data_dict)
        self.seq_pred[video_id].extend(pred_instances_list)

    def compute_metrics(self, results: list = None) -> dict:
        self._lazy_init()
        logger: MMLogger = MMLogger.get_current_instance()
        eval_results = dict()

        if self.format_only:
            logger.info("Only formatting results to the official format.")
            return eval_results

        resfile_path = self.outfile_prefix

        default_eval_config = teta.config.get_default_eval_config()
        default_eval_config["PRINT_ONLY_COMBINED"] = True
        default_eval_config["DISPLAY_LESS_PROGRESS"] = True
        default_eval_config["OUTPUT_TEM_RAW_DATA"] = True
        default_eval_config["NUM_PARALLEL_CORES"] = 8
        default_dataset_config = teta.config.get_default_dataset_config()
        default_dataset_config["TRACKERS_TO_EVAL"] = ["MASA"]
        default_dataset_config["GT_FOLDER"] = self.ann_file
        default_dataset_config["OUTPUT_FOLDER"] = resfile_path
        default_dataset_config["TRACKER_SUB_FOLDER"] = os.path.join(
            resfile_path, "tao_track.json"
        )

        evaluator = teta.Evaluator(default_eval_config)
        dataset_list = [teta.datasets.TAO(default_dataset_config)]
        print("Overall classes performance")
        eval_results, _ = evaluator.evaluate(dataset_list, [teta.metrics.TETA()])

        if self.open_vocabulary:
            eval_results_path = os.path.join(
                resfile_path, "MASA", "teta_summary_results.pth"
            )
            eval_res = pickle.load(open(eval_results_path, "rb"))

            base_class_synset = set(
                [
                    c["name"]
                    for c in self.coco.dataset["categories"]
                    if c["frequency"] != "r"
                ]
            )
            novel_class_synset = set(
                [
                    c["name"]
                    for c in self.coco.dataset["categories"]
                    if c["frequency"] == "r"
                ]
            )

            self.compute_teta_on_ovsetup(
                eval_res, base_class_synset, novel_class_synset
            )

        return eval_results

    def evaluate(self, size: int = 1) -> dict:
        self._lazy_init()
        logger: MMLogger = MMLogger.get_current_instance()

        logger.info(f"Wait for all processes to complete prediction.")
        barrier()

        logger.info(f"Start gathering tracking results.")
        gathered_seq_info = all_gather_object(dict(self.seq_pred))

        if is_main_process():
            all_seq_pred = dict()
            for _seq_info in gathered_seq_info:
                all_seq_pred.update(_seq_info)
            all_seq_pred = self.compute_global_track_id(all_seq_pred)

            all_seq_pred_json = list(chain.from_iterable(all_seq_pred.values()))

            if self.tcc and all_seq_pred_json:
                all_seq_pred_json = self.majority_vote(all_seq_pred_json)

            result_files_path = f"{self.outfile_prefix}/tao_track.json"

            logger.info(f"Saving json pred file into {result_files_path}")
            mmengine.dump(all_seq_pred_json, result_files_path)

            logger.info(f"Start evaluation")
            _metrics = self.compute_metrics()

            if self.prefix:
                _metrics = {"/".join((self.prefix, k)): v for k, v in _metrics.items()}
            metrics = [_metrics]
        else:
            metrics = [None]

        broadcast_object_list(metrics)
        self.seq_pred.clear()

        return metrics[0]

    def compute_global_track_id(self, all_seq_pred):
        max_track_id = 0
        # Sort by video_id to ensure deterministic order
        for video_id in sorted(all_seq_pred.keys()):
            seq_pred = all_seq_pred[video_id]
            if not seq_pred:
                continue

            # Find the max local track ID in the current video
            max_local_id = 0
            for frame_pred in seq_pred:
                if frame_pred["track_id"] > max_local_id:
                    max_local_id = frame_pred["track_id"]

            # Apply the offset
            for frame_pred in seq_pred:
                frame_pred["track_id"] += max_track_id

            # Update the global max track ID for the next video
            max_track_id += max_local_id + 1
        return all_seq_pred

    def majority_vote(self, prediction):
        df_pred_res = pd.DataFrame(prediction)
        groued_df_pred_res = df_pred_res.groupby("track_id")
        class_by_majority_count_res = []
        for _, group in tqdm.tqdm(groued_df_pred_res):
            cid = group["category_id"].mode()[0]
            group["category_id"] = cid
            class_by_majority_count_res.extend(group.to_dict("records"))
        return class_by_majority_count_res

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def compute_teta_on_ovsetup(self, teta_res, base_class_names, novel_class_names):
        if "COMBINED_SEQ" in teta_res:
            teta_res = teta_res["COMBINED_SEQ"]

        frequent_teta = []
        rare_teta = []
        for key in teta_res:
            if key in base_class_names:
                frequent_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))
            elif key in novel_class_names:
                rare_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))

        print("Base and Novel classes performance")
        print(
            "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "TETA50:", "TETA", "LocA", "AssocA", "ClsA", "LocRe", "LocPr",
                "AssocRe", "AssocPr", "ClsRe", "ClsPr"
            )
        )

        if frequent_teta:
            freq_teta_mean = np.mean(np.stack(frequent_teta), axis=0)
            print("{:<10} ".format("Base"), end="")
            print(*["{:<10.3f}".format(num) for num in freq_teta_mean])
        else:
            print("No Base classes to evaluate!")
            freq_teta_mean = None
        if rare_teta:
            rare_teta_mean = np.mean(np.stack(rare_teta), axis=0)
            print("{:<10} ".format("Novel"), end="")
            print(*["{:<10.3f}".format(num) for num in rare_teta_mean])
        else:
            print("No Novel classes to evaluate!")
            rare_teta_mean = None

        return freq_teta_mean, rare_teta_mean
