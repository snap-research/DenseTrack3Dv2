import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import glob

import pickle
import numpy as np

import torch
import torch.nn.functional as F

import argparse
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.cloud_opt.delta_optimizer import DeltaPointCloudOptimizer
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.io_utils import save_colmap_cameras, save_colmap_images, storePly
from densetrack3d.datasets.custom_data import read_data, read_data_with_depthcrafter

from sfm.sfm_utils import make_pairs, get_3D_model_from_scene
from densetrack3d.datasets.tum import TUMDataset

from densetrack3d.datasets.tartanair_dataset import TartanAirDataset
from densetrack3d.datasets.utils import collate_fn
from densetrack3d.evaluation.core.traj_metrics import eval_metrics

import roma

from sfm.traj_utils import load_traj

def prepare_single_pair(pair, pair_id, data_dict, keyframe_id_list, keyframe_masks, view1, view2, pred1, pred2, resolution=(384,512)):
    view1['img'].append(pair[0])
    view2['img'].append(pair[1])
    view1['idx'].append(pair_id[0])
    view2['idx'].append(pair_id[1])
    view1['instance'].append(str(pair_id[0]))
    view2['instance'].append(str(pair_id[1]))
    view1['true_shape'].append(torch.tensor(resolution))
    view2['true_shape'].append(torch.tensor(resolution))

    # pred1['pts3d'].append(data_dict[str(pair_id[0])]['tracks_3d'])

    keyframe1 = str(keyframe_id_list[pair_id[0]])
    keyframe2 = str(keyframe_id_list[pair_id[1]])

    mask1 = keyframe_masks[pair_id[0]]
    mask2 = keyframe_masks[pair_id[1]]

    view1['dynamic_mask'].append(mask1.float())
    view2['dynamic_mask'].append(mask2.float())

    pts3d_1 = data_dict[keyframe1]['trajs_3d'][keyframe1][:,:3].astype(np.float32)
    conf_1 = data_dict[keyframe1]['conf'][keyframe2].astype(np.float32) # NOTE trick here, conf of view1 is the visibility at view 2
    pts3d_1 = torch.from_numpy(pts3d_1).float().reshape(*resolution,3)
    conf_1 = torch.from_numpy(conf_1).float().reshape(*resolution)

    conf_1 = 1 + conf_1.exp()
    pts3d_2_at_1 = data_dict[keyframe2]['trajs_3d'][keyframe1][:,:3].astype(np.float32)


    conf_2_at_1 = data_dict[keyframe2]['conf'][keyframe1].astype(np.float32)
    pts3d_2_at_1 = torch.from_numpy(pts3d_2_at_1).float().reshape(*resolution,3)
    conf_2_at_1 = torch.from_numpy(conf_2_at_1).float().reshape(*resolution)
    conf_2_at_1 = 1.0 + conf_2_at_1.exp()

    pts2d_1 = data_dict[keyframe1]['trajs_uv'][keyframe1][:,:2].astype(np.float32)
    pts2d_1_at_2 = data_dict[keyframe1]['trajs_uv'][keyframe2][:,:2].astype(np.float32)

    pts2d_1 = torch.from_numpy(pts2d_1).float().reshape(*resolution,2)
    pts2d_1_at_2 = torch.from_numpy(pts2d_1_at_2).float().reshape(*resolution,2)
    flow2d_12 = pts2d_1_at_2 - pts2d_1

    visib_1_at_2 = data_dict[keyframe1]['conf'][keyframe2].astype(np.float32)
    visib_1_at_2 = (torch.from_numpy(visib_1_at_2) > 0.9).float().reshape(*resolution)

    pts2d_2 = data_dict[keyframe2]['trajs_uv'][keyframe2][:,:2].astype(np.float32)
    pts2d_2_at_1 = data_dict[keyframe2]['trajs_uv'][keyframe1][:,:2].astype(np.float32)

    pts2d_2 = torch.from_numpy(pts2d_2).float().reshape(*resolution,2)
    pts2d_2_at_1 = torch.from_numpy(pts2d_2_at_1).float().reshape(*resolution,2)
    flow2d_21 = pts2d_2_at_1 - pts2d_2

    visib_2_at_1 = data_dict[keyframe2]['conf'][keyframe1].astype(np.float32)
    visib_2_at_1 = (torch.from_numpy(visib_2_at_1) > 0.9).float().reshape(*resolution)

    pred1['pts3d'].append(pts3d_1)
    pred1['conf'].append(conf_1)
    pred1['flow2d'].append(flow2d_12)
    pred1['visib'].append(visib_1_at_2)

    pred2['pts3d_in_other_view'].append(pts3d_2_at_1)
    pred2['conf'].append(conf_2_at_1)
    pred2['flow2d'].append(flow2d_21)
    pred2['visib'].append(visib_2_at_1)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="demo_data/rollerblade", help="demo video path")
    parser.add_argument("--output_path", type=str, default="results/demo", help="output path")


    return parser

if __name__ == "__main__":

    test_dataset = TUMDataset(
        data_root="tum",
        use_metric_depth=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"TartanAir dataset loaded, num samples: {len(test_dataloader)}")

    avg_ate, avg_rpe_trans, avg_rpe_rot = [], [], []

    for ind, sample in enumerate(test_dataloader):


        # start_f = 0
        # end_f = 60
        # chunk_size = 8
        # chunk_start = 0

        vid_name = sample.seq_name[0]


        # data_path = f"vis_results/custom_data/tape3d_pairwise/{vid_name}_pred_dict_pairwise.pkl"

        track_dir = "results/densetrack_tum/pairwise_stride2"
        data_path = os.path.join(track_dir, f"{vid_name}_pred_dict_pairwise.pkl")
        with open(data_path, 'rb') as handle:
            data_dict = pickle.load(handle)

        # video, videodepth, videomask = read_data("vis_results/custom_data", vid_name, resize_384x512=True)
        video = sample.video.cuda()
        videodepth = sample.videodepth.cuda()
        videomask = torch.zeros_like(videodepth).bool()

        resized_hw = data_dict['0']['conf']['0'].shape[0]
        resized_w = 512
        assert resized_hw % resized_w == 0
        resized_h = int(resized_hw / 512 )

        resolution = (resized_h, resized_w)
        if video.shape[-2] != resolution[0] or video.shape[-1] != resolution[1]:
            B, T = video.shape[:2]
            video = F.interpolate(video.flatten(0,1), size=resolution, mode='bilinear', align_corners=False).reshape(B,T,3,*resolution)
            if videomask is not None:
                videomask = F.interpolate(videomask.float().flatten(0,1), size=resolution, mode='nearest').reshape(B,T,1,*resolution).bool()


        start_f = 0
        end_f = 100
        total_keyframe_id_list = list(data_dict.keys())[start_f:end_f]
        total_keyframes = len(total_keyframe_id_list)
        keyframe_id_list = total_keyframe_id_list
        keyframe_id_list = [int(k) for k in keyframe_id_list]

        print("keyframe_id_list", keyframe_id_list)

        keyframe_imgs = video[0, keyframe_id_list]
        keyframe_imgs = (keyframe_imgs / 255.0) * 2.0 - 1.0

        keyframe_masks = videomask[0, keyframe_id_list]
        keyframe_masks = keyframe_masks.squeeze(1)

        num_keyframes = len(keyframe_id_list)
        pairs_id = make_pairs(keyframe_imgs)

        pairs = []
        for (i,j) in pairs_id:
            pairs.append((keyframe_imgs[i], keyframe_imgs[j]))

        print("Num pairs:", len(pairs))

        view1 = {
            'img': [],
            'dynamic_mask': [],
            'true_shape': [],
            'idx': [],
            'instance': []
        }

        view2 = {
            'img': [],
            'dynamic_mask': [],
            'true_shape': [],
            'idx': [],
            'instance': []
        }

        pred1 = {
            'pts3d': [],
            'conf': [],
            # 'flow3d': [],
            'flow2d': [],
            'flow2d': [],
            "visib": []
        }

        pred2 = {
            'pts3d_in_other_view': [],
            'conf': [],
            # 'flow3d': [],
            'flow2d': [],
            "visib": []
        }

        # breakpoint()

        for pair, pair_id in zip(pairs, pairs_id):
            prepare_single_pair(pair, pair_id, data_dict, keyframe_id_list, keyframe_masks, view1, view2, pred1, pred2, resolution=resolution)

        view1['img'] = torch.stack(view1['img']).cuda()
        view2['img'] = torch.stack(view2['img']).cuda()
        view1['dynamic_mask'] = torch.stack(view1['dynamic_mask']).cuda()
        view2['dynamic_mask'] = torch.stack(view2['dynamic_mask']).cuda()
        view1['true_shape'] = torch.stack(view1['true_shape'], dim=0).cuda()
        view2['true_shape'] = torch.stack(view2['true_shape'], dim=0).cuda()

        pred1['pts3d'] = torch.stack(pred1['pts3d']).cuda()
        pred1['conf'] = torch.stack(pred1['conf']).cuda()
        pred1['flow2d'] = torch.stack(pred1['flow2d']).cuda()
        pred1['visib'] = torch.stack(pred1['visib']).cuda()

        pred2['pts3d_in_other_view'] = torch.stack(pred2['pts3d_in_other_view']).cuda()
        pred2['conf'] = torch.stack(pred2['conf']).cuda()
        pred2['flow2d'] = torch.stack(pred2['flow2d']).cuda()
        pred2['visib'] = torch.stack(pred2['visib']).cuda()

        device ="cuda"

        num_total_iter = 300
        schedule = "linear"
        lr = 0.01
        log_dir = f'logdirs/deltav2_pose_est_dust3r/tum/{vid_name}'
        # save_path = f'logdirs/dust3r_pose_mpi_sintel/{vid_name}'
        # os.makedirs(save_path, exist_ok=True)

        # os.makedirs(log_dir, exist_ok=True)
        optim_kw = {
            'verbose': True, 
            'min_conf_thr': 0.5, 
            'log_dir': log_dir, 
            'num_total_iter': num_total_iter, 
            'use_self_dynamic_mask': False
        }
        scene = DeltaPointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)

        loss = scene.compute_global_alignment(
            init='mst', 
            niter=num_total_iter, 
            schedule=schedule, 
            lr=lr,
            use_delta_init=True,
        )
                    
        rgbimg = scene.imgs
        focals = scene.get_focals().cpu().detach().numpy()
        cams2world = scene.get_im_poses().cpu().detach().numpy()

        pts3d, _ = scene.get_pts3d(return_pose=True, fuse=True)
        pts3d = to_numpy(pts3d)
        msk = to_numpy(scene.get_masks())

        pts = np.stack(pts3d, axis=0)
        col = np.stack(rgbimg, axis=0)
        mask = np.stack(msk, axis=0)

        pts_color = np.concatenate([pts, col], axis=-1) # t n 6

        save_dict = {
            'pts_color': pts_color,
            'pts_mask': mask,
            'focals': focals,
            'cams2world': cams2world,
            'keyframe_id': keyframe_id_list,
            # 'pts3d': pts3d,
            # 'msk': msk
        }

        with open(os.path.join(log_dir, f'{vid_name}.pkl'), 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        gt_traj = load_traj(
            f"tum/{vid_name}/groundtruth_90.txt",
            "tum",
            skip=0,
            stride=1,
        )

        # os.makedirs("debug/eval_traj_sintel_depthcrafter", exist_ok=True)

        cams2world_torch = torch.from_numpy(cams2world).float()
        rot_mats = cams2world_torch[:, :3, :3]
        unit_quat_wxyz = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(rot_mats))
        trans = cams2world_torch[:, :3, 3]
        eval_poses = torch.cat([trans, unit_quat_wxyz], axis=1)
        
        # q0, q1 = roma.random_unitquat(size=10), roma.random_unitquat(size=10)
        # steps = torch.linspace(0, 1.0, 5)
        # q_interpolated = roma.utils.unitquat_slerp(q0, q1, steps)

        # breakpoint()
        # NOTE interpolate here
        eval_poses_all = []
        allframe_id_list = list(range(0, len(gt_traj[0])))

        last_keyframe_id = None
        for frame_id in allframe_id_list:
            if frame_id // 2 >= len(eval_poses):
                eval_poses_all.append(eval_poses[last_keyframe_id])
            elif frame_id in keyframe_id_list:
                eval_poses_all.append(eval_poses[frame_id // 2])
                last_keyframe_id = frame_id // 2
            else:
                if frame_id+1 >= len(gt_traj[0]) or (frame_id+1)//2 >= len(eval_poses):
                    eval_poses_all.append(eval_poses[(frame_id-1)//2])
                else:
                    prev_pose = eval_poses[(frame_id-1)//2]
                    post_pose = eval_poses[(frame_id+1)//2]
                    
                    rot_to_avg = torch.stack([prev_pose[3:], post_pose[3:]], dim=0) # 2 x 4
                    rot_to_avg = roma.quat_wxyz_to_xyzw(rot_to_avg)
                    avg_rot = roma.utils.unitquat_slerp(rot_to_avg[0], rot_to_avg[1], torch.tensor([0.5]))
                    avg_rot = roma.quat_xyzw_to_wxyz(avg_rot)[0]

                    # w_i = torch.tensor([0.5, 0.5])
                    # rot_to_avg = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(rot_to_avg))
                    # avg_rot = roma.special_procrustes(torch.sum(w_i[:,None, None] * rot_to_avg, dim=0)) # weighted average.
                    # avg_rot = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(avg_rot))
                    avg_trans = (prev_pose[:3] + post_pose[:3]) * 0.5

                    # breakpoint()
                    avg_pose = torch.cat([avg_trans, avg_rot], axis=0)
                    eval_poses_all.append(avg_pose)
        eval_poses_all = torch.stack(eval_poses_all, dim=0)
        eval_poses_all = eval_poses_all.numpy()
        
        # timestamps = np.arange(0, len(eval_poses_all))[:, None]
        timestamps = gt_traj[1]
        pred_traj = list((eval_poses_all, timestamps))

        # breakpoint()
        try:
            ate, rpe_trans, rpe_rot = eval_metrics(
                pred_traj,
                gt_traj=gt_traj,
                seq=vid_name,
                filename=os.path.join(log_dir, f"eval_metrics_{vid_name}.txt"),
            )
        except:
            ate, rpe_trans, rpe_rot = np.nan, np.nan, np.nan

        avg_ate.append(ate)
        avg_rpe_trans.append(rpe_trans)
        avg_rpe_rot.append(rpe_rot)


        print(f"Done {sample.seq_name[0]}")
        # if ind ==1:
        #     break

    print(f"Num failed: {np.isnan(np.asarray(avg_ate)).sum()} / {len(avg_ate)}")
    
    avg_ate = np.nanmean(np.asarray(avg_ate))
    avg_rpe_trans = np.nanmean(np.asarray(avg_rpe_trans))
    avg_rpe_rot = np.nanmean(np.asarray(avg_rpe_rot))

    print("Average ATE trans:", avg_ate)
    print("Average RPE trans:", avg_rpe_trans)
    print("Average RPE rot:", avg_rpe_rot)


    eval_files = [f for f in os.listdir(log_dir) if f.endswith(".txt")]

    new_avg_ate, new_avg_rpe_trans, new_avg_rpe_rot = [], [], []
    for eval_file in eval_files:
        with open(os.path.join(log_dir, eval_file), 'r') as f:
            lines = f.readlines()

            # print(lines[9])
            ate = float(lines[9].split('\t')[1].strip())
            rpe_rot = float(lines[20].split('\t')[1].strip())
            rpe_trans = float(lines[31].split('\t')[1].strip())


        
            new_avg_ate.append(ate)
            new_avg_rpe_trans.append(rpe_trans)
            new_avg_rpe_rot.append(rpe_rot)

    
    new_avg_ate = np.nanmean(np.asarray(new_avg_ate))
    new_avg_rpe_trans = np.nanmean(np.asarray(new_avg_rpe_trans))
    new_avg_rpe_rot = np.nanmean(np.asarray(new_avg_rpe_rot))

    print("Average ATE trans:", new_avg_ate)
    print("Average RPE trans:", new_avg_rpe_trans)
    print("Average RPE rot:", new_avg_rpe_rot)