import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import json
import time
from dataclasses import dataclass, field

import hydra
import numpy as np
import ray
import torch
from densetrack3d.datasets.tapvid2d_dataset import TapVid2DDataset
from densetrack3d.datasets.utils import collate_fn
from densetrack3d.evaluation.core.evaluator import Evaluator

# from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2

from densetrack3d.models.evaluation_predictor.evaluation_predictor import EvaluationPredictor
from densetrack3d.utils.io import load_checkpoint
from omegaconf import OmegaConf

TAPVID2D_DIR = None


@dataclass(eq=False)
class DefaultConfig:
    # Directory where all outputs of the experiment will be saved.
    exp_dir: str = "./results/densetrack3d_tapvid2d"

    # Name of the dataset to be used for the evaluation.
    # dataset_name: str = "tapvid_davis_first"
    # dataset_type: str = "rgb_stacking" # "rgb_stacking" # "davis"
    queried_first: bool = True
    # dataset_name: str = "lsfodyssey"
    # The root directory of the dataset.
    dataset_root: str = TAPVID2D_DIR

    checkpoint: str = "checkpoints/densetrack3dv2.pth"

    # EvaluationPredictor parameters
    # The size (N) of the support grid used in the predictor.
    # The total number of points is (N*N).
    grid_size: tuple = (5, 5)
    # The size (N) of the local support grid.
    local_grid_size: int = 8
    # A flag indicating whether to evaluate one ground truth point at a time.
    single_point: bool = False
    # The number of iterative updates for each sliding window.
    n_iters: int = 6

    seed: int = 0
    gpu_idx: int = 0

    # Override hydra's working directory to current working dir,
    # also disable storing the .hydra logs:
    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},
            "output_subdir": None,
        }
    )


@ray.remote(num_gpus=1)
class RayPredictor:
    def __init__(
        self,
        worker_id: int = 0,
        cfg: DefaultConfig = None,
    ):
        self.worker_id = worker_id
        self.cfg = cfg

    def run_eval(self, split):
        cfg = self.cfg

        os.makedirs(cfg.exp_dir, exist_ok=True)

        evaluator = Evaluator(cfg.exp_dir)

        model = DenseTrack3DV2(
            stride=4,
            window_len=16,
            add_space_attn=True,
            num_virtual_tracks=64,
            model_resolution=(384, 512),
            coarse_to_fine_dense=True
        )

        state_dict = load_checkpoint(cfg.checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        predictor = EvaluationPredictor(
            model,
            grid_size=5,
            local_grid_size=cfg.local_grid_size,
            single_point=cfg.single_point,
            n_iters=cfg.n_iters,
            # use_disp=False,
        )
        predictor = predictor.eval().cuda()

        test_dataset = TapVid2DDataset(
            dataset_type=split,
            data_root=cfg.dataset_root,
            queried_first=cfg.queried_first,
            read_from_s3=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Timing and conducting the evaluation

        start = time.time()
        evaluate_result = evaluator.evaluate_sequence(
            predictor,
            test_dataloader,
            dataset_name=f"tapvid_{split}_first",
            is_sparse=True,
            verbose=True,
            visualize_every=-1,
        )
        end = time.time()
        print(f"Time taken for evaluation on {split} split: {end - start:.3f} seconds")

        saved_evaluate_result = {}
        saved_evaluate_result["time"] = end - start
        saved_evaluate_result["avg"] = evaluate_result["avg"]
        print("Evaluate_result (avg)", saved_evaluate_result["avg"])

        result_file = os.path.join(cfg.exp_dir, f"result_eval_{split}.json")
        print(f"Dumping eval results to {result_file}.")
        with open(result_file, "w") as f:
            json.dump(saved_evaluate_result, f, indent=4)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config_eval", node=DefaultConfig)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = DefaultConfig()

    # splits = ["davis"]
    splits = ["kinetics"]

    num_gpus = min(len(splits), torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs")

    ray.init()
    predictors = [RayPredictor.remote(i, cfg) for i in range(num_gpus)]
    tasks = [predictor.run_eval.remote(split) for predictor, split in zip(predictors, splits)]

    ray.get(tasks)
    ray.shutdown()
