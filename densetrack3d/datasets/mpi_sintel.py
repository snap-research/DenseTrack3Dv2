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

from densetrack3d.datasets.utils import DeltaData
# from densetrack3d.datasets.utils import CoTrackerData
from densetrack3d.models.geometry_utils import least_square_align


TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"



def depth_read_sintel(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))

    return depth
    # NOTE mask invalid values
    # max_depth_valid = depth[depth < 100].max()
    # min_depth_valid = depth[depth > 0].min()

    # depth[depth > max_depth_valid] = max_depth_valid
    # depth[depth < min_depth_valid] = min_depth_valid
    # return depth, max_depth_valid, min_depth_valid

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



def flow_read_sintel(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

class MPISintelDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root="default", 
            use_gt_depth=False,
            # datatype="pstudio", 
            # crop_size=256, 
            # small=False,
            # use_metric_depth=True,
            # split="minival"
        ):
        super(MPISintelDataset, self).__init__()

        self.use_gt_depth = use_gt_depth
        self.data_root = data_root
        self.calib_root = data_root.replace("final", "camdata_left")
        self.depth_root = data_root.replace("final", "depth")
        # self.scene_names = sorted(list("alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3".split(" ")))
        self.scene_names = sorted(["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"])
        if not self.use_gt_depth:

            with open(f'{data_root}/training/depthcrafter_depth.pkl', 'rb') as handle:
                self.predicted_depth = pickle.load(handle)

    def __getitem__(self, index):
        scene_name = self.scene_names[index]

        scene_path = os.path.join(self.data_root, scene_name)
        calib_path = os.path.join(self.calib_root, scene_name)
        depth_path = os.path.join(self.depth_root, scene_name)

        img_names = sorted([n for n in os.listdir(scene_path) if n.endswith(".png")])

        video = []
        gt_videodepth = []
        intrinsics = []
        for img_name in img_names:
            image = cv2.imread(os.path.join(scene_path, img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            gt_depth = depth_read_sintel(os.path.join(depth_path, img_name.replace(".png", ".dpt")))

            max_depth_valid = gt_depth[gt_depth < 100].max()
            min_depth_valid = gt_depth[gt_depth > 0].min()

            gt_depth[gt_depth > max_depth_valid] = max_depth_valid
            gt_depth[gt_depth < min_depth_valid] = min_depth_valid
            

            camfile = os.path.join(calib_path, img_name.replace(".png", ".cam"))
            K, _ = cam_read_sintel(camfile)
            intrinsic = K[:3,:3]
            # fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            # calib = [fx, fy, cx, cy]
            # if len(calib) > 4:
            #     image = cv2.undistort(image, K, calib[4:])

            video.append(image)
            gt_videodepth.append(gt_depth)
            intrinsics.append(intrinsic)
        
        video = np.stack(video)
        gt_videodepth = np.stack(gt_videodepth)
        intrinsics = np.stack(intrinsics)

        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        gt_videodepth = torch.from_numpy(gt_videodepth).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        max_depth_valid = torch.tensor(max_depth_valid)
        min_depth_valid = torch.tensor(min_depth_valid)

        if self.use_gt_depth:
            videodepth = gt_videodepth
        else:
            videodisp = torch.from_numpy(self.predicted_depth[scene_name]).float()
            videodepth = least_square_align(gt_videodepth, videodisp, return_align_scalar=False, query_frame=0)

            videodepth[videodepth > max_depth_valid] = max_depth_valid
            videodepth[videodepth < min_depth_valid] = min_depth_valid

            videodepth = videodepth.unsqueeze(1)
        
        data = DeltaData(
            video=video,
            videodepth=videodepth,
            intrs=intrinsics,
            seq_name=scene_name,
        )


        return data

    def __len__(self):
        return len(self.scene_names)
    

class MPISintelFlowDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root="MPI-Sintel/training/final", 
            use_gt_depth=False,
            # datatype="pstudio", 
            # crop_size=256, 
            # small=False,
            # use_metric_depth=True,
            # split="minival"
        ):
        super(MPISintelFlowDataset, self).__init__()

        self.use_gt_depth = use_gt_depth
        self.data_root = data_root
        # self.calib_root = data_root.replace("final", "camdata_left")
        self.depth_root = data_root.replace("final", "depth")
        self.flow_root = data_root.replace("final", "flow")
        self.scene_names = sorted(["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"])
        # self.scene_names = sorted(os.listdir(self.data_root))
        self.scene_names = sorted(["cave_2", "cave_4"])
        
        # if not self.use_gt_depth:

        with open(f'{data_root}/training/depthcrafter_depth.pkl', 'rb') as handle:
            self.predicted_depth = pickle.load(handle)

        self.image_list, self.extra_info, self.flow_list = [], [], []
        for scene_name in self.scene_names:
            image_list = sorted(glob.glob(os.path.join(self.data_root, scene_name, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene_name, i) ] # scene and frame_id

                self.flow_list += sorted(glob.glob(os.path.join(self.flow_root, scene_name, '*.flo')))

        # print(self.depth_root)
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        info = self.extra_info[index]
        img_names = self.image_list[index]
        flow_name = self.flow_list[index]

        scene_name = info[0]
        frame_id = info[1]
        # scene_name = self.scene_names[index]

        # scene_path = os.path.join(self.data_root, scene_name)
        # calib_path = os.path.join(self.calib_root, scene_name)
        depth_path = os.path.join(self.depth_root, scene_name)


        # print(depth_path, scene_name, frame_id)

        video = []
        # gt_videodepth = []
        for img_name in img_names:

            # print(os.path.join(scene_path, img_name), frame_id, img_name)
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            video.append(image)
        
        video = np.stack(video)
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()


        # base_img_name = os.path.basename(img_name)
        gt_depth_firstframe = depth_read_sintel(os.path.join(depth_path, "frame_0001.dpt"))
        max_depth_valid = gt_depth_firstframe[gt_depth_firstframe < 100].max()
        min_depth_valid = gt_depth_firstframe[gt_depth_firstframe > 0].min()

        gt_depth_firstframe[gt_depth_firstframe > max_depth_valid] = max_depth_valid
        gt_depth_firstframe[gt_depth_firstframe < min_depth_valid] = min_depth_valid

        gt_depth_firstframe = torch.from_numpy(gt_depth_firstframe).float()
        disp_firstframe = torch.from_numpy(self.predicted_depth[scene_name][0]).float()
        _, s, t = least_square_align(gt_depth_firstframe[None], disp_firstframe[None], return_align_scalar=True, query_frame=0)
        # gt_videodepth = np.stack(gt_videodepth)

        videodisp = torch.from_numpy(self.predicted_depth[scene_name][frame_id:frame_id+2]).float()
        videodepth = 1 / torch.clamp((videodisp * s + t), 1e-6, 1e6)
        videodepth = videodepth.unsqueeze(1)
        # gt_videodepth = torch.from_numpy(gt_videodepth).float()

        max_depth_valid = torch.tensor(max_depth_valid)
        min_depth_valid = torch.tensor(min_depth_valid)

        gt_flow = flow_read_sintel(flow_name)
        gt_flow = np.array(gt_flow).astype(np.float32)
        gt_flow = torch.from_numpy(gt_flow).float()
        gt_flow[torch.isnan(gt_flow)] = 0
        gt_flow[gt_flow.abs() > 1e9] = 0
        valid = (gt_flow[...,0].abs() < 1000) & (gt_flow[...,1].abs() < 1000)



        
        data = DeltaData(
            video=video,
            videodepth=videodepth,
            seq_name=f"{scene_name}_{frame_id:05d}",
            flow=gt_flow,
            flow_valid=valid
        )


        return data

    

