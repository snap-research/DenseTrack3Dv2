import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import json
import os
from dataclasses import dataclass, field

import hydra
import numpy as np
import pickle

import torch
from omegaconf import OmegaConf

from tqdm import tqdm

import torch.nn.functional as F

import argparse


from densetrack3d.models.geometry_utils import least_square_align
from densetrack3d.datasets.custom_data import read_data, read_data_with_depthcrafter
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.predictor.dense_predictor import DensePredictor3D
from densetrack3d.models.densetrack3d.densetrack3d_v11_pyramid import DenseTrack3DV11Pyramid
from densetrack3d.utils.depthcrafter_utils import read_video
from densetrack3d.datasets.utils import collate_fn

from densetrack3d.datasets.tum import TUMDataset

# from densetrack3d.datasets.mpi_sintel import MPISintelDataset

BASE_DIR = os.getcwd()
device = torch.device("cuda")


if __name__ == "__main__":

    maximum_T = 101
    stride = 2
    chunk_size = 16

    save_dir = "results/densetrack_tum/pairwise_stride2"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint: str = "logdirs/densetrack3d_v11_pyramid/model_densetrack3d_final.pth"
    print("Create DenseTrack3D model")
    model = DenseTrack3DV11Pyramid(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        coarse_to_fine_dense=True
    )

    print(f"Load checkpoint from {checkpoint}")
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)

    predictor = DensePredictor3D(model=model)
    predictor = predictor.eval().cuda()

    test_dataset = TUMDataset(
        data_root="datasets/tum",
        use_metric_depth=False
    )

    # Creating the DataLoader object
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    for ind, sample in enumerate(test_dataloader):
        
        # if ind % 2 != 1: continue
        # if ind < 4: continue
        vid_name = sample.seq_name[0]

        # if vid_name not in ['market_6']: continue
        # if os.path.exists(os.path.join(save_dir, f'{vid_name}_pred_dict_pairwise.pkl')):
        #     continue
        

        video = sample.video.cuda()
        videodepth = sample.videodepth.cuda()


        max_T = min(maximum_T, video.shape[1])

        key_inds = list(range(0, max_T, stride))

        video = video[:, :max_T]
        videodepth = videodepth[:, :max_T]
        total_dict = {}
        for ind in key_inds:
            print(f"Densely tracking points at frame {ind}")

            fw_dict, bw_dict = None, None
            if ind < video.shape[1]:
                fw_video = video[:, ind:min(ind+chunk_size, video.shape[1])]
                fw_videodepth = videodepth[:, ind:min(ind+chunk_size, video.shape[1])]
                fw_out_dict = predictor(
                    fw_video,
                    fw_videodepth,
                    grid_query_frame=0,
                )

                fw_dict = {
                    'trajs_3d': fw_out_dict["trajs_3d_dict"]["coords"][0].cpu().numpy(),
                    'trajs_uv': fw_out_dict["trajs_uv"][0].cpu().numpy(),
                    'conf': fw_out_dict["conf"][0].cpu().numpy(),
                }

            if ind > 0:
                bw_video = video[:, max(0, ind-chunk_size):ind+1].flip(1)
                bw_videodepth = videodepth[:, max(0, ind-chunk_size):ind+1].flip(1)
                bw_out_dict = predictor(
                    bw_video,
                    bw_videodepth,
                    grid_query_frame=0,
                )

                bw_dict = {
                    'trajs_3d': bw_out_dict["trajs_3d_dict"]["coords"][0].cpu().numpy(),
                    'trajs_uv': bw_out_dict["trajs_uv"][0].cpu().numpy(),
                    'conf': bw_out_dict["conf"][0].cpu().numpy(),
                }

            strided_save_dict = {
                'trajs_3d': {},
                'trajs_uv': {},
                'conf': {},
            }

            if fw_dict is not None:
                for k in fw_dict.keys():
                    for tstep in range(0, chunk_size, stride): # 0,2,4,6
                        if tstep >= len(fw_dict[k]): break

                        strided_save_dict[k][str(ind+tstep)] = fw_dict[k][tstep]

            if bw_dict is not None:
                for k in bw_dict.keys():
                    for tstep in range(stride, chunk_size, stride):
                        if tstep >= len(bw_dict[k]): break

                        strided_save_dict[k][str(ind-tstep)] = bw_dict[k][tstep]

            total_dict[str(ind)] = strided_save_dict

        save_path = os.path.join(save_dir, f'{vid_name}_pred_dict_pairwise.pkl')
        print(f"Saving to {save_path}")
        with open(save_path, 'wb') as handle:
            pickle.dump(total_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

