import os
import io
import glob
import torch
import torch.nn.functional as F

import pickle
import numpy as np
import mediapy as media
import cv2
from scipy.spatial.transform import Rotation
from PIL import Image
from typing import Mapping, Tuple, Union

# from cotracker.datasets.utils import CoTrackerData
from densetrack3d.datasets.utils import DeltaData


TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"

def lsq(depth, disp):
    T, H, W = disp.shape

    if depth.shape[1] != H or depth.shape[2] != W:
        depth = F.interpolate(depth.float().unsqueeze(1), (H, W), mode='nearest').squeeze(1)

    inv_depth = 1 / torch.clamp(depth, 1e-6, 1e6)
    
    # NOTE only align first frame
    x = disp[0].clone().reshape(-1)
    y = inv_depth[0].clone().reshape(-1)

    A = torch.stack([x, torch.ones_like(x)], dim=-1) # N, 2

    # A = np.vstack([x, np.ones(len(x))]).T

    s, t = torch.linalg.lstsq(A, y, rcond=None)[0]

    aligned_disp = disp * s + t

    aligned_depth = 1 / torch.clamp(aligned_disp, 1e-6, 1e6)

    return aligned_depth.reshape(T, H, W), s, t 

def cam_read_sintel(filename):
    """Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    M = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
    N = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
    return M, N

class TUMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root="datasets/tum", 
            use_metric_depth=False,
            rgb_folder="rgb_90"
            # datatype="pstudio", 
            # crop_size=256, 
            # small=False,
            # use_metric_depth=True,
            # split="minival"
        ):
        super(TUMDataset, self).__init__()

        self.use_metric_depth = use_metric_depth
        self.data_root = data_root
        self.rgb_folder = rgb_folder

        # if "90" in self.rgb_folder:
        #     self.depth_folder = "depth_90"
        # else:
        #     self.depth_folder = "depth"
        # self.calib_root = data_root.replace("final", "camdata_left")
        # self.depth_root = data_root.replace("final", "depth")
        # self.scene_names = sorted(list("alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3".split(" ")))
        self.scene_names = sorted([s for s in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, s))])

        
        # sorted(["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"])
        # if not self.use_gt_depth:

        if os.path.exists(os.path.join(self.data_root, "depthcrafter_depth.pkl")):
            with open(os.path.join(self.data_root, "depthcrafter_depth.pkl"), "rb") as handle:
                self.depthcrafter_depth = pickle.load(handle)
        else:
            self.depthcrafter_depth = None

        if os.path.exists(os.path.join(self.data_root, "unidepth_depth.pkl")):
            with open(os.path.join(self.data_root, "unidepth_depth.pkl"), "rb") as handle:
                self.unidepth_depth = pickle.load(handle)
        else:
            self.unidepth_depth = None

    def __getitem__(self, index):
        scene_name = self.scene_names[index]

        rgb_path = os.path.join(self.data_root, scene_name, self.rgb_folder)
        # depth_path = os.path.join(self.data_root, scene_name, self.depth_folder)

        img_names = sorted([n for n in os.listdir(rgb_path) if n.endswith(".png")])
        # depth_names = sorted([n for n in os.listdir(depth_path) if n.endswith(".png")])

        video = []

        for i, img_name in enumerate(img_names):
            image = cv2.imread(os.path.join(rgb_path, img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video.append(image)
        
        video = np.stack(video)
        videodepth = self.unidepth_depth[scene_name].astype(np.float32)

        max_depth_valid = videodepth[videodepth < 100].max()
        min_depth_valid = videodepth[videodepth > 0].min()

        videodepth[videodepth > max_depth_valid] = max_depth_valid
        videodepth[videodepth < min_depth_valid] = min_depth_valid

        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        videodepth = torch.from_numpy(videodepth).float()

        max_depth_valid = torch.tensor(max_depth_valid)
        min_depth_valid = torch.tensor(min_depth_valid)

        if self.use_metric_depth:
            videodepth = videodepth.unsqueeze(1)
        else:
            videodisp = torch.from_numpy(self.depthcrafter_depth[scene_name]).float()
            videodepth, _, _ = lsq(videodepth, videodisp)

            videodepth[videodepth > max_depth_valid] = max_depth_valid
            videodepth[videodepth < min_depth_valid] = min_depth_valid

            videodepth = videodepth.unsqueeze(1)
        
        data = DeltaData(
            video=video,
            videodepth=videodepth,
            seq_name=scene_name,
        )


        return data

    def __len__(self):
        return len(self.scene_names)

