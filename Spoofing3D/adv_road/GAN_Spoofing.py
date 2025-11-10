# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp
import numpy as np
import cv2
import random

import mmcv
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import collate, scatter
from mmcv.ops import box_iou_rotated

from mmdet import __version__ as mmdet_version

from Spoofing3D.adv_utils.common_utils import rotate_points_along_z
from Spoofing3D.adv_utils.dcgan import DCGAN_D_CustomAspectRatio, weights_init, DCGAN_G_CustomAspectRatio, SceneSet
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model, init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from matplotlib.path import Path

import lpips

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--type', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--resume_netD', default='', type=str, help='resume from this discriminator checkpoint')
    parser.add_argument('--resume_netG', default='', type=str, help='resume from this generator checkpoint')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def mmlabDeNormalize(img):
    from mmcv.image.photometric import imdenormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_bgr = True
    img = img.permute(1,2,0).numpy()
    img = imdenormalize(img, mean, std, to_bgr)
    return img

def visulize_3dbox_to_cam(img, gt_coner, sample_info, cam_name, post_rot, post_tran, img_h, img_w):
    #img (H,W,3) BGR
    #gt_corner (N, 8, 3)
    #sample_info dict 信息字典
    #cam_name img是哪个相机
    #post_rot (3,3)由图像增强所带来的旋转矩阵
    #post_tran （3，3）由图像增强所带来的平移矩阵
    #imgh 图像高
    #imgw 图像宽
    from pyquaternion import Quaternion
    lidar2lidarego = np.eye(4, dtype=np.float32)
    lidar2lidarego[:3, :3] = Quaternion(sample_info['lidar2ego_rotation']).rotation_matrix
    lidar2lidarego[:3, 3] = sample_info['lidar2ego_translation']
    lidar2lidarego = torch.from_numpy(lidar2lidarego)

    lidarego2global = np.eye(4, dtype=np.float32)
    lidarego2global[:3, :3] = Quaternion(sample_info['ego2global_rotation']).rotation_matrix
    lidarego2global[:3, 3] = sample_info['ego2global_translation']
    lidarego2global = torch.from_numpy(lidarego2global)

    cam2camego = np.eye(4, dtype=np.float32)
    cam2camego[:3, :3] = Quaternion(sample_info['cams'][cam_name]['sensor2ego_rotation']).rotation_matrix
    cam2camego[:3, 3] = sample_info['cams'][cam_name]['sensor2ego_translation']
    cam2camego = torch.from_numpy(cam2camego)

    camego2global = np.eye(4, dtype=np.float32)
    camego2global[:3, :3] = Quaternion(sample_info['cams'][cam_name]['ego2global_rotation']).rotation_matrix
    camego2global[:3, 3] = sample_info['cams'][cam_name]['ego2global_translation']
    camego2global = torch.from_numpy(camego2global)

    cam2img = np.eye(4, dtype=np.float32)
    cam2img = torch.from_numpy(cam2img)
    cam2img[:3, :3] = torch.from_numpy(sample_info['cams'][cam_name]['cam_intrinsic'])

    lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(lidarego2global.matmul(lidar2lidarego))
    lidar2img = cam2img.matmul(lidar2cam)

    gt_coner = gt_coner.view(-1,3)
    gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
    gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)

    gt_coner = gt_coner.view(-1, 8 ,3)


    def is_in_img (p1,p2,img_h,img_w):
        if p1[2] < 1 or p2[2] < 1:
            return False
        flag1 = (p1[0] >= 0) & (p1[0] < img_w) & (p1[1] >= 0) & (p1[1] < img_h) & (p1[2] < 60) & (p1[2] > 1)
        flag2 = (p2[0] >= 0) & (p2[0] < img_w) & (p2[1] >= 0) & (p2[1] < img_h) & (p2[2] < 60) & (p2[2] > 1)
        return flag1 or flag2

    for obj_i in range(len(gt_coner)):
        corner = gt_coner[obj_i] #(8,3)
        connect = [[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],[0, 4], [1, 5], [2, 6], [3, 7]]
        for line_i in connect:
            p1 = corner[line_i[0]]
            p2 = corner[line_i[1]]
            if is_in_img(p1,p2,img_h,img_w):
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255), thickness=1)
    #cv2.imwrite('demo.png',img)
    return img

def visulize_3dbox_to_cam_multiFrame(img, gt_coner,cam2lidar_r,cam2lidar_t,intrin, post_rot, post_tran, img_h, img_w):
    #img (H,W,3) BGR
    #gt_corner (N, 8, 3)
    #sample_info dict 信息字典
    #cam_name img是哪个相机
    #post_rot (3,3)由图像增强所带来的旋转矩阵
    #post_tran （3，3）由图像增强所带来的平移矩阵
    #imgh 图像高
    #imgw 图像宽

    cam2lidar = torch.eye(4)
    cam2lidar[:3,:3] = cam2lidar_r
    cam2lidar[:3,3] = cam2lidar_t
    lidar2cam = torch.inverse(cam2lidar)

    cam2img = torch.eye(4)
    cam2img[:3,:3] = intrin

    lidar2img = cam2img.matmul(lidar2cam)


    gt_coner = gt_coner.view(-1,3)
    gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
    gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)

    gt_coner = gt_coner.view(-1, 8 ,3)

    def is_in_img(p1, p2, img_h, img_w):
        if p1[2] < 1 or p2[2] < 1:
            return False
        flag1 = (p1[0] >= 0) & (p1[0] < img_w) & (p1[1] >= 0) & (p1[1] < img_h) & (p1[2] < 60) & (p1[2] > 1)
        flag2 = (p2[0] >= 0) & (p2[0] < img_w) & (p2[1] >= 0) & (p2[1] < img_h) & (p2[2] < 60) & (p2[2] > 1)
        return flag1 or flag2

    for obj_i in range(len(gt_coner)):
        corner = gt_coner[obj_i] #(8,3)
        connect = [[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],[0, 4], [1, 5], [2, 6], [3, 7]]
        for line_i in connect:
            p1 = corner[line_i[0]]
            p2 = corner[line_i[1]]
            if is_in_img(p1,p2,img_h,img_w):
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255), thickness=1)#(64,128,255)
    #cv2.imwrite('demo.png',img)
    return img

def draw_box_from_batch(img,batch_inputs,frame_i,cam_i,img_h=256,img_w=704):
    cam2lidar = torch.eye(4)
    cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
    cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
    lidar2cam = torch.inverse(cam2lidar)

    cam2img = torch.eye(4)
    cam2img[:3, :3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
    lidar2img = cam2img.matmul(lidar2cam)

    gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners
    post_rot = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu()
    post_tran = batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()

    gt_coner = gt_coner.view(-1, 3)
    gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
    gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
    gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)
    gt_coner = gt_coner.view(-1, 8, 3)

    def is_in_img (p1,p2,img_h,img_w):
        if p1[2] < 1 or p2[2] < 1:
            return False
        flag1 = (p1[0] >= 0) & (p1[0] < img_w) & (p1[1] >= 0) & (p1[1] < img_h) & (p1[2] < 60) & (p1[2] > 1)
        flag2 = (p2[0] >= 0) & (p2[0] < img_w) & (p2[1] >= 0) & (p2[1] < img_h) & (p2[2] < 60) & (p2[2] > 1)
        return flag1 or flag2

    for obj_i in range(len(gt_coner)):
        corner = gt_coner[obj_i] #(8,3)
        connect = [[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],[0, 4], [1, 5], [2, 6], [3, 7]]
        for line_i in connect:
            p1 = corner[line_i[0]]
            p2 = corner[line_i[1]]
            if is_in_img(p1,p2,img_h,img_w):
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255), thickness=1)
    #cv2.imwrite('demo.png',img)
    return img


#将图像中的gt bbox mask掉
def maskGT_put_poster_on_batch_inputs(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], is_bilinear=False, mask_aug=False):
    #leaning_poster (m,3,200,300)
    use_poster_idx=0

    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0].size(0)
    '''mask gt bbox'''
    for frame_i in range(batchsize):
        for cam in cam_names:
            cam_i = camname_idx[cam]
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3, :3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu(), \
                                  batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()

            gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners #(G,8,3)
            gt_coner = gt_coner.view(-1, 3)
            gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
            gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            gt_coner = gt_coner.view(-1, 8, 3)

            for gt_i in range(len(gt_coner)):
                corner_3d = gt_coner[gt_i,:,:] #(8,3) [u,v,z]
                #if corner_3d[:,2].min() <=1 : continue
                if (corner_3d[:,2]>1).sum() <= 2 : continue
                corner_3d = corner_3d[corner_3d[:,2] > 1]

                xmin, ymin = corner_3d[:,0].min(), corner_3d[:,1].min()
                xmax, ymax = corner_3d[:,0].max(), corner_3d[:,1].max()

                if xmin > img_w-1 or ymin > img_h - 1 or xmax <= 0 or ymax <= 0 : continue
                xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(img_w-1, int(xmax)), min(img_h-1, int(ymax))
                batch_inputs['img_inputs'][0][frame_i,cam_i,:,ymin:ymax,xmin:xmax] = ((torch.Tensor([[0.5,0.5,0.5]])- torch.from_numpy(mean)) / torch.from_numpy(std)).cuda().permute(1,0).unsqueeze(-1)


    '''put poster on image'''
    for frame_i in range(batchsize):
        num_obj_in_frame = len(batch_inputs['gt_bboxes_3d'][frame_i].tensor)

        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (5 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (0 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1. : continue #防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)

            '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3,:3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
            post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu(), batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()
            img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            img_corner = img_corner[:, :2]  # (4,2)

            '''求解图像区域内的所有像素点坐标***************************************************************************************'''
            path = Path(img_corner.numpy())
            x, y = np.mgrid[:img_w, :img_h]
            points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
            mask = path.contains_points(points)
            path_points = points[np.where(mask)]  # (Nin,2) [x,y]
            img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
            if len(img_inner_points) <= 200: continue #如果在图像上的poster像素点少于200，就不要这个实例了

            '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
            img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
            R = torch.inverse(lidar2img[:3, :3].T)
            T = lidar2img[:3, 3]

            fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
            fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
            C = fz / fm  # (Nin)
            img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                              (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                              C.unsqueeze(-1)], dim=1)
            lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

            '''找到每个3D点在poster上的颜色索引,并put到原图上'''
            lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
            lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
            lidar_inner_points[:, 0] += l / 2.
            lidar_inner_points[:, 1] += w / 2.

            if mask_aug:
                if np.random.random() < 0.4:
                    leaning_poster_clone = leaning_poster
                else:
                    leaning_poster_clone = leaning_poster.clone()
                    for p_i in range(leaning_poster.size(0)):
                        bound_pixel = 50
                        start_w = np.random.choice(poster_w-bound_pixel)
                        start_l = np.random.choice(poster_l-bound_pixel)
                        end_w = min(start_w+50, poster_w)
                        end_l = min(start_l+50, poster_l)
                        leaning_poster_clone[p_i,:,start_w:end_w, start_l:end_l] = 0

            else:
                leaning_poster_clone = leaning_poster


            if is_bilinear:
                index_l = torch.clip((lidar_inner_points[:, 0] / l)*2-1, min=-1, max=1)
                index_w = torch.clip(((w - lidar_inner_points[:, 1]) / w)*2-1, min=-1, max=1)
                grid = torch.cat([index_l.unsqueeze(-1), index_w.unsqueeze(-1)], dim=1).unsqueeze(0).unsqueeze(0) #(1,1,Nin,2)
                selected_color = torch.nn.functional.grid_sample(leaning_poster_clone[use_poster_idx].unsqueeze(0), grid.cuda(), mode='bilinear', align_corners=True) #(1,3,1,Nin)
                selected_color = selected_color.squeeze().permute(1,0)

            else :
                index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                selected_color = leaning_poster_clone[use_poster_idx, :, index_w, index_l].T #(Nin, 3) bgr 0~1 gpu
            use_poster_idx+=1

            contrast = round(random.uniform(0.8, 1.0), 10)
            brightness = round(random.uniform(-0.15, 0.1), 10)
            selected_color = selected_color * contrast + brightness
            selected_color[selected_color > 1] = 1
            selected_color[selected_color < 0] = 0

            selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
            batch_inputs['img_inputs'][0][frame_i, cam_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

            batch_inputs['gt_bboxes_3d'][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
            gt_label = batch_inputs['gt_labels_3d'][frame_i]
            batch_inputs['gt_labels_3d'][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])
        #只保留spoofer的gt信息
        batch_inputs['gt_bboxes_3d'][frame_i].tensor = batch_inputs['gt_bboxes_3d'][frame_i].tensor[num_obj_in_frame:,:]
        batch_inputs['gt_labels_3d'][frame_i] = batch_inputs['gt_labels_3d'][frame_i][num_obj_in_frame:].long()

#测试时候放到scatter后的batch inputs里
def put_poster_on_batch_inputs_eval(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], mask_aug=False,use_next_poster=False):
    use_poster_idx = 0

    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4, 2)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0][0].size(0)
    for frame_i in range(batchsize):
        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][0][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (5*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (0*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][0][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1.: continue  # 防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
            #print(poster_corner)

            '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][0][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][0][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3,:3] = batch_inputs['img_inputs'][0][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
            post_rot, post_tran = batch_inputs['img_inputs'][0][4][frame_i][cam_i].cpu(), batch_inputs['img_inputs'][0][5][frame_i][cam_i].cpu()
            img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            img_corner = img_corner[:, :2]  # (4,2)

            '''求解图像区域内的所有像素点坐标***************************************************************************************'''
            path = Path(img_corner.numpy())
            x, y = np.mgrid[:img_w, :img_h]
            points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
            mask = path.contains_points(points)
            path_points = points[np.where(mask)]  # (Nin,2) [x,y]
            img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
            if len(img_inner_points) <= 200: continue #如果在图像上的poster像素点少于200，就不要这个实例了

            '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
            img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
            R = torch.inverse(lidar2img[:3, :3].T)
            T = lidar2img[:3, 3]

            fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
            fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
            C = fz / fm  # (Nin)
            img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                              (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                              C.unsqueeze(-1)], dim=1)
            lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

            if mask_aug:
                leaning_poster_clone = leaning_poster.clone()
                for p_i in range(len(leaning_poster_clone)):
                    bound_pixel = 50
                    start_w = np.random.choice(poster_w - bound_pixel)
                    start_l = np.random.choice(poster_l - bound_pixel)
                    end_w = min(start_w + 50, poster_w)
                    end_l = min(start_l + 50, poster_l)
                    leaning_poster_clone[p_i,:,start_w:end_w, start_l:end_l] = 0
            else:
                leaning_poster_clone = leaning_poster.clone()


            '''找到每个3D点在poster上的颜色索引,并put到原图上'''
            lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
            lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
            lidar_inner_points[:, 0] += l / 2.
            lidar_inner_points[:, 1] += w / 2.

            index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
            index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
            selected_color = leaning_poster_clone[use_poster_idx,:,index_w, index_l].T #(Nin, 3) bgr 0~1 gpu
            if use_next_poster:
                use_poster_idx += 1

            #contrast = round(random.uniform(0.7, 1.0), 10)
            #brightness = round(random.uniform(-0.3, 0.2), 10)
            #selected_color = selected_color * contrast + brightness
            #selected_color[selected_color > 1] = 1
            #selected_color[selected_color < 0] = 0

            selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
            batch_inputs['img_inputs'][0][0][frame_i, cam_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

            batch_inputs['gt_bboxes_3d'][0][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
            gt_label = batch_inputs['gt_labels_3d'][0][frame_i]
            batch_inputs['gt_labels_3d'][0][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])

#第二阶段训练时，可指定放置的location
def maskGT_put_poster_on_batch_inputs_v2(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], mask_aug=False,location_dict=None,use_next_poster=True):
    use_poster_idx = 0
    if location_dict is None:
        is_sampling_loaction = True
        location_dict = []
    else:
        is_sampling_loaction = False


    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0].size(0)
    '''mask gt bbox'''
    for frame_i in range(batchsize):
        for cam in cam_names:
            cam_i = camname_idx[cam]
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3, :3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu(), \
                                  batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()

            gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners  # (G,8,3)
            gt_coner = gt_coner.view(-1, 3)
            gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
            gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            gt_coner = gt_coner.view(-1, 8, 3)

            for gt_i in range(len(gt_coner)):
                corner_3d = gt_coner[gt_i, :, :]  # (8,3) [u,v,z]
                # if corner_3d[:,2].min() <=1 : continue
                if (corner_3d[:, 2] > 1).sum() <= 2: continue
                corner_3d = corner_3d[corner_3d[:, 2] > 1]

                xmin, ymin = corner_3d[:, 0].min(), corner_3d[:, 1].min()
                xmax, ymax = corner_3d[:, 0].max(), corner_3d[:, 1].max()

                if xmin > img_w - 1 or ymin > img_h - 1 or xmax <= 0 or ymax <= 0: continue
                xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(img_w - 1, int(xmax)), min(img_h - 1,
                                                                                                              int(ymax))
                batch_inputs['img_inputs'][0][frame_i, cam_i, :, ymin:ymax, xmin:xmax] = (
                            (torch.Tensor([[0.5, 0.5, 0.5]]) - torch.from_numpy(mean)) / torch.from_numpy(
                        std)).cuda().permute(1, 0).unsqueeze(-1)

    for frame_i in range(batchsize):
        num_obj_in_frame = len(batch_inputs['gt_bboxes_3d'][frame_i].tensor)

        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            if is_sampling_loaction:
                search_flag = 0
                for _ in range(max_search_num):
                    r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                    an = (2 * np.random.rand() - 1) * (5*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                    yaw = (2 * np.random.rand() - 1) * (0 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                    cx = r * np.cos(an)
                    cy = r * np.sin(an)

                    fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                    fake_box = torch.from_numpy(fake_box)

                    bev_iou = box_iou_rotated(fake_box, gt_bev)
                    if len(gt_bev) == 0:
                        break
                    if bev_iou.max() == 0:
                        break
                    search_flag += 1
                if search_flag == max_search_num:
                    location_dict.append(dict(valid=False))
                    continue
                else:
                    location_dict.append(dict(valid=True, r=r, an=an, yaw=yaw))
            else:
                if location_dict[0]['valid']:
                    r, an, yaw = location_dict[0]['r'], location_dict[0]['an'], location_dict[0]['yaw']
                    cx = r * np.cos(an)
                    cy = r * np.sin(an)

                    fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                    fake_box = torch.from_numpy(fake_box)

                    location_dict.pop(0)
                else:
                    location_dict.pop(0)
                    continue


            #car_z = gt_box[batch_inputs['gt_labels_3d'][0][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1.: continue  # 防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
            #print(poster_corner)

            '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3,:3] = batch_inputs['img_inputs'][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
            post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][cam_i].cpu(), batch_inputs['img_inputs'][5][frame_i][cam_i].cpu()
            img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            img_corner = img_corner[:, :2]  # (4,2)

            '''求解图像区域内的所有像素点坐标***************************************************************************************'''
            path = Path(img_corner.numpy())
            x, y = np.mgrid[:img_w, :img_h]
            points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
            mask = path.contains_points(points)
            path_points = points[np.where(mask)]  # (Nin,2) [x,y]
            img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
            if len(img_inner_points) <= 200: continue #如果在图像上的poster像素点少于200，就不要这个实例了

            '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
            img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
            R = torch.inverse(lidar2img[:3, :3].T)
            T = lidar2img[:3, 3]

            fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
            fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
            C = fz / fm  # (Nin)
            img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                              (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                              C.unsqueeze(-1)], dim=1)
            lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

            if mask_aug:
                leaning_poster_clone = leaning_poster.clone()
                for p_i in range(len(leaning_poster_clone)):
                    bound_pixel = 50
                    start_w = np.random.choice(poster_w - bound_pixel)
                    start_l = np.random.choice(poster_l - bound_pixel)
                    end_w = min(start_w + 50, poster_w)
                    end_l = min(start_l + 50, poster_l)
                    leaning_poster_clone[p_i,:,start_w:end_w, start_l:end_l] = 0
            else:
                leaning_poster_clone = leaning_poster.clone()


            '''找到每个3D点在poster上的颜色索引,并put到原图上'''
            lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
            lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
            lidar_inner_points[:, 0] += l / 2.
            lidar_inner_points[:, 1] += w / 2.

            index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
            index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
            selected_color = leaning_poster_clone[use_poster_idx,:,index_w, index_l].T #(Nin, 3) bgr 0~1 gpu
            if use_next_poster:
                use_poster_idx += 1

            #contrast = round(random.uniform(0.7, 1.0), 10)
            #brightness = round(random.uniform(-0.3, 0.2), 10)
            #selected_color = selected_color * contrast + brightness
            #selected_color[selected_color > 1] = 1
            #selected_color[selected_color < 0] = 0

            selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
            batch_inputs['img_inputs'][0][frame_i, cam_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

            batch_inputs['gt_bboxes_3d'][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
            gt_label = batch_inputs['gt_labels_3d'][frame_i]
            batch_inputs['gt_labels_3d'][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])

        # 只保留spoofer的gt信息
        batch_inputs['gt_bboxes_3d'][frame_i].tensor = batch_inputs['gt_bboxes_3d'][frame_i].tensor[num_obj_in_frame:, :]
        batch_inputs['gt_labels_3d'][frame_i] = batch_inputs['gt_labels_3d'][frame_i][num_obj_in_frame:].long()

    return location_dict


#测试时候放到scatter后的batch inputs里 跨相机
def put_poster_on_batch_inputs_eval_cross_cam(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK']):
    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -160, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -20}
    sample_range = (6, 12)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (3.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[:2]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0][0].size(0)
    for frame_i in range(batchsize):
        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][0][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (48*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (10*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][0][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1.: continue  # 防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
            #print(poster_corner)

            '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
            cam2lidar = torch.eye(4)
            cam2lidar[:3, :3] = batch_inputs['img_inputs'][0][1][frame_i][cam_i].cpu()
            cam2lidar[:3, 3] = batch_inputs['img_inputs'][0][2][frame_i][cam_i].cpu()
            lidar2cam = torch.inverse(cam2lidar)

            cam2img = torch.eye(4)
            cam2img[:3,:3] = batch_inputs['img_inputs'][0][3][frame_i][cam_i].cpu()
            lidar2img = cam2img.matmul(lidar2cam)
            img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
            post_rot, post_tran = batch_inputs['img_inputs'][0][4][frame_i][cam_i].cpu(), batch_inputs['img_inputs'][0][5][frame_i][cam_i].cpu()
            img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
            img_corner = img_corner[:, :2]  # (4,2)

            '''求解图像区域内的所有像素点坐标***************************************************************************************'''
            path = Path(img_corner.numpy())
            x, y = np.mgrid[:img_w, :img_h]
            points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
            mask = path.contains_points(points)
            path_points = points[np.where(mask)]  # (Nin,2) [x,y]
            img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
            if len(img_inner_points) <= 200: continue #如果在图像上的poster像素点少于200，就不要这个实例了

            '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
            img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
            R = torch.inverse(lidar2img[:3, :3].T)
            T = lidar2img[:3, 3]

            fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
            fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
            C = fz / fm  # (Nin)
            img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                              (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                              C.unsqueeze(-1)], dim=1)
            lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)


            '''找到每个3D点在poster上的颜色索引,并put到原图上'''
            lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
            lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
            lidar_inner_points[:, 0] += l / 2.
            lidar_inner_points[:, 1] += w / 2.

            index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
            index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
            selected_color = leaning_poster[index_w, index_l, :] #(Nin, 3) bgr 0~1 gpu

            #contrast = round(random.uniform(0.7, 1.0), 10)
            #brightness = round(random.uniform(-0.3, 0.2), 10)
            #selected_color = selected_color * contrast + brightness
            #selected_color[selected_color > 1] = 1
            #selected_color[selected_color < 0] = 0

            selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
            batch_inputs['img_inputs'][0][0][frame_i, cam_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

            batch_inputs['gt_bboxes_3d'][0][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
            gt_label = batch_inputs['gt_labels_3d'][0][frame_i]
            batch_inputs['gt_labels_3d'][0][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])

            for ccam_i, ccam in enumerate(cam_names):
                if ccam == spoofcam: continue
                # 求解对应于图像中的四个角点*************************************************'''
                cam2lidar = torch.eye(4)
                cam2lidar[:3, :3] = batch_inputs['img_inputs'][0][1][frame_i][ccam_i].cpu()
                cam2lidar[:3, 3] = batch_inputs['img_inputs'][0][2][frame_i][ccam_i].cpu()
                lidar2cam = torch.inverse(cam2lidar)
                cam2img = torch.eye(4)
                cam2img[:3, :3] = batch_inputs['img_inputs'][0][3][frame_i][ccam_i].cpu()
                lidar2img = cam2img.matmul(lidar2cam)
                img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
                post_rot, post_tran = batch_inputs['img_inputs'][0][4][frame_i][ccam_i].cpu(), \
                                      batch_inputs['img_inputs'][0][5][frame_i][ccam_i].cpu()
                img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
                if (img_corner[:, 2] > 0).sum() < 4: continue
                img_corner = img_corner[:, :2]  # (4,2)
                # 求解图像区域内的所有像素点坐标***************************************************************************************'''
                path = Path(img_corner.numpy())
                x, y = np.mgrid[:img_w, :img_h]
                points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
                mask = path.contains_points(points)
                path_points = points[np.where(mask)]  # (Nin,2) [x,y]
                img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
                if len(img_inner_points) == 0: continue
                '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
                img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(
                    torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
                R = torch.inverse(lidar2img[:3, :3].T)
                T = lidar2img[:3, 3]

                fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
                fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
                C = fz / fm  # (Nin)
                img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                                  (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                                  C.unsqueeze(-1)], dim=1)
                lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)
                '''找到每个3D点在poster上的颜色索引,并put到原图上'''
                lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
                lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0),-1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
                lidar_inner_points[:, 0] += l / 2.
                lidar_inner_points[:, 1] += w / 2.

                index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                selected_color = leaning_poster[index_w, index_l, :]  # (Nin, 3) bgr 0~1 gpu

                selected_color = (selected_color - torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()  # (Nin, 3) 归一化
                batch_inputs['img_inputs'][0][0][frame_i, ccam_i, :, img_inner_points[:, 1],img_inner_points[:, 0]] = selected_color.T


#for BEVDet4D
def maskGT_put_poster_on_batch_inputs_4D(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], is_bilinear=False, num_adj=8,use_next_poster=True):
    #leaning_poster (m,3,200,300)
    use_poster_idx=0

    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0].size(0)
    '''mask gt bbox'''
    for frame_i in range(batchsize):
        for time_i in range(num_adj+1):
            for cam in cam_names:
                cam_i = camname_idx[cam]
                cam2lidar = torch.eye(4)
                cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][6*time_i+cam_i].cpu()
                cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][6*time_i+cam_i].cpu()
                lidar2cam = torch.inverse(cam2lidar)

                cam2img = torch.eye(4)
                cam2img[:3, :3] = batch_inputs['img_inputs'][3][frame_i][6*time_i+cam_i].cpu()
                lidar2img = cam2img.matmul(lidar2cam)
                post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][6*time_i+cam_i].cpu(), \
                                      batch_inputs['img_inputs'][5][frame_i][6*time_i+cam_i].cpu()

                gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners #(G,8,3)
                gt_coner = gt_coner.view(-1, 3)
                gt_coner = gt_coner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                gt_coner = torch.cat([gt_coner[:, :2] / gt_coner[:, 2:3], gt_coner[:, 2:3]], 1)
                gt_coner = gt_coner.matmul(post_rot.T) + post_tran.unsqueeze(0)
                gt_coner = gt_coner.view(-1, 8, 3)

                for gt_i in range(len(gt_coner)):
                    corner_3d = gt_coner[gt_i,:,:] #(8,3) [u,v,z]
                    #if corner_3d[:,2].min() <=1 : continue
                    if (corner_3d[:,2]>1).sum() <= 2 : continue
                    corner_3d = corner_3d[corner_3d[:,2] > 1]

                    xmin, ymin = corner_3d[:,0].min(), corner_3d[:,1].min()
                    xmax, ymax = corner_3d[:,0].max(), corner_3d[:,1].max()

                    if xmin > img_w-1 or ymin > img_h - 1 or xmax <= 0 or ymax <= 0 : continue
                    xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(img_w-1, int(xmax)), min(img_h-1, int(ymax))
                    batch_inputs['img_inputs'][0][frame_i, (num_adj+1)*cam_i+time_i, :,ymin:ymax,xmin:xmax] = ((torch.Tensor([[0.5,0.5,0.5]])- torch.from_numpy(mean)) / torch.from_numpy(std)).cuda().permute(1,0).unsqueeze(-1)


    '''put poster on image'''
    for frame_i in range(batchsize):
        num_obj_in_frame = len(batch_inputs['gt_bboxes_3d'][frame_i].tensor)

        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (5 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (0 * np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1. : continue #防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)

            for time_i in range(num_adj+1):
                '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
                cam2lidar = torch.eye(4)
                cam2lidar[:3, :3] = batch_inputs['img_inputs'][1][frame_i][6*time_i+cam_i].cpu()
                cam2lidar[:3, 3] = batch_inputs['img_inputs'][2][frame_i][6*time_i+cam_i].cpu()
                lidar2cam = torch.inverse(cam2lidar)

                cam2img = torch.eye(4)
                cam2img[:3,:3] = batch_inputs['img_inputs'][3][frame_i][6*time_i+cam_i].cpu()
                lidar2img = cam2img.matmul(lidar2cam)
                img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                if (img_corner[:, 2] < 1).sum(): continue
                img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
                post_rot, post_tran = batch_inputs['img_inputs'][4][frame_i][6*time_i+cam_i].cpu(), batch_inputs['img_inputs'][5][frame_i][6*time_i+cam_i].cpu()
                img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
                img_corner = img_corner[:, :2]  # (4,2)

                '''求解图像区域内的所有像素点坐标***************************************************************************************'''
                path = Path(img_corner.numpy())
                x, y = np.mgrid[:img_w, :img_h]
                points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
                mask = path.contains_points(points)
                path_points = points[np.where(mask)]  # (Nin,2) [x,y]
                img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
                if time_i == 0 and len(img_inner_points) <= 200: break
                if time_i != 0 and len(img_inner_points) <= 10 : continue

                '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
                img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
                R = torch.inverse(lidar2img[:3, :3].T)
                T = lidar2img[:3, 3]

                fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
                fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
                C = fz / fm  # (Nin)
                img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                                  (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                                  C.unsqueeze(-1)], dim=1)
                lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

                '''找到每个3D点在poster上的颜色索引,并put到原图上'''
                lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
                lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
                lidar_inner_points[:, 0] += l / 2.
                lidar_inner_points[:, 1] += w / 2.

                if is_bilinear:
                    index_l = torch.clip((lidar_inner_points[:, 0] / l)*2-1, min=-1, max=1)
                    index_w = torch.clip(((w - lidar_inner_points[:, 1]) / w)*2-1, min=-1, max=1)
                    grid = torch.cat([index_l.unsqueeze(-1), index_w.unsqueeze(-1)], dim=1).unsqueeze(0).unsqueeze(0) #(1,1,Nin,2)
                    selected_color = torch.nn.functional.grid_sample(leaning_poster[use_poster_idx].unsqueeze(0), grid.cuda(), mode='bilinear', align_corners=True) #(1,3,1,Nin)
                    selected_color = selected_color.squeeze().permute(1,0)

                else :
                    index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                    index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                    selected_color = leaning_poster[use_poster_idx, :, index_w, index_l].T #(Nin, 3) bgr 0~1 gpu

                contrast = round(random.uniform(0.8, 1.0), 10)
                brightness = round(random.uniform(-0.15, 0.1), 10)
                selected_color = selected_color * contrast + brightness
                selected_color[selected_color > 1] = 1
                selected_color[selected_color < 0] = 0

                selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
                batch_inputs['img_inputs'][0][frame_i, (num_adj+1)*cam_i+time_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

                if time_i == 0:
                    batch_inputs['gt_bboxes_3d'][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
                    gt_label = batch_inputs['gt_labels_3d'][frame_i]
                    batch_inputs['gt_labels_3d'][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])
            if use_next_poster:
                use_poster_idx += 1
        #只保留spoofer的gt信息
        batch_inputs['gt_bboxes_3d'][frame_i].tensor = batch_inputs['gt_bboxes_3d'][frame_i].tensor[num_obj_in_frame:,:]
        batch_inputs['gt_labels_3d'][frame_i] = batch_inputs['gt_labels_3d'][frame_i][num_obj_in_frame:].long()

def put_poster_on_batch_inputs_eval_4D(leaning_poster, batch_inputs, spoof_cams=['CAM_FRONT', 'CAM_BACK'], num_adj=8,use_next_poster=True, only_apply_cur_frame=False):
    use_poster_idx = 0

    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32)/255  # bgr下
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32)/255
    img_h, img_w = 256, 704
    camname_idx = {'CAM_FRONT_LEFT': 0, 'CAM_FRONT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_BACK_LEFT': 3, 'CAM_BACK': 4, 'CAM_BACK_RIGHT': 5}
    camcenter_angle = {'CAM_FRONT_LEFT': 145, 'CAM_FRONT': 90, 'CAM_FRONT_RIGHT': 35, 'CAM_BACK_LEFT': -1, 'CAM_BACK': -90, 'CAM_BACK_RIGHT': -1}
    sample_range = (7, 10)
    default_lwh = (4., 1.8, 1.6)
    physical_lw = (4.0, 2.0)
    max_search_num = 20

    poster_w, poster_l = leaning_poster.size()[2:]
    delta_l, delta_w = physical_lw[0] / poster_l, physical_lw[1]  / poster_w
    # *****************************************************************************************************************************************************#
    batchsize = batch_inputs['img_inputs'][0][0].size(0)
    for frame_i in range(batchsize):
        for spoofcam in spoof_cams:
            cam_i = camname_idx[spoofcam]
            gt_box = batch_inputs['gt_bboxes_3d'][0][frame_i].tensor  # (N,9)
            gt_bev = gt_box[:, [0, 1, 3, 4, 6]]  # (N,5) [cx,cy,h,w,theta]
            '''确定poster的3D位置******************************************************************************'''
            search_flag = 0
            for _ in range(max_search_num):
                r = np.random.rand() * (sample_range[1] - sample_range[0]) + sample_range[0]
                an = (2 * np.random.rand() - 1) * (5*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.  # 加的常数应该与相机本身的角度有关,保证poster落在选定的相机内
                cx = r * np.cos(an)
                cy = r * np.sin(an)
                yaw = (2 * np.random.rand() - 1) * (0*np.pi / 180.) + camcenter_angle[spoofcam] * np.pi / 180.
                fake_box = np.array([[cx, cy, default_lwh[0], default_lwh[1], yaw]]).astype(np.float32)
                fake_box = torch.from_numpy(fake_box)

                bev_iou = box_iou_rotated(fake_box, gt_bev)
                if len(gt_bev) == 0:
                    break
                if bev_iou.max() == 0:
                    break
                search_flag += 1
            if search_flag == max_search_num: continue

            #car_z = gt_box[batch_inputs['gt_labels_3d'][0][frame_i] == 0]
            car_z = gt_box
            if len(car_z) == 0:
                z_bottle = -2.
            else:
                min_idx = torch.argmin(torch.sum((car_z[:, :2] - fake_box[:, :2]) ** 2, dim=1))
                z_bottle = car_z[min_idx, 2]
            if z_bottle > -1.: continue  # 防止飘在空中的情况
            fake_3d_box = torch.Tensor([[fake_box[0, 0], fake_box[0, 1], z_bottle, default_lwh[0], default_lwh[1], default_lwh[2], fake_box[0, 4], 0, 0]])
            #print(fake_3d_box)

            '''求解海报四个角点在3D LiDAR系下的坐标****************************************************************'''
            l, w = physical_lw[0], physical_lw[1]
            poster_corner = torch.Tensor([[l / 2, w / 2, z_bottle],
                                          [l / 2, -w / 2, z_bottle],
                                          [-l / 2, -w / 2, z_bottle],
                                          [-l / 2, w / 2, z_bottle]]).unsqueeze(0)  # (1,4,3)

            poster_corner = rotate_points_along_z(poster_corner, torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (4,3)
            poster_corner[:, :2] += fake_3d_box[:, :2]  # (4,3)
            #print(poster_corner)

            for time_i in range(num_adj + 1):
                '''求解对应于图像中的四个角点,暂未约束超出图像边界的情况*************************************************'''
                cam2lidar = torch.eye(4)
                cam2lidar[:3, :3] = batch_inputs['img_inputs'][0][1][frame_i][6*time_i+cam_i].cpu()
                cam2lidar[:3, 3] = batch_inputs['img_inputs'][0][2][frame_i][6*time_i+cam_i].cpu()
                lidar2cam = torch.inverse(cam2lidar)

                cam2img = torch.eye(4)
                cam2img[:3,:3] = batch_inputs['img_inputs'][0][3][frame_i][6*time_i+cam_i].cpu()
                lidar2img = cam2img.matmul(lidar2cam)
                img_corner = poster_corner.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                if (img_corner[:, 2] < 1).sum(): continue
                img_corner = torch.cat([img_corner[:, :2] / img_corner[:, 2:3], img_corner[:, 2:3]], 1)
                post_rot, post_tran = batch_inputs['img_inputs'][0][4][frame_i][6*time_i+cam_i].cpu(), batch_inputs['img_inputs'][0][5][frame_i][6*time_i+cam_i].cpu()
                img_corner = img_corner.matmul(post_rot.T) + post_tran.unsqueeze(0)
                img_corner = img_corner[:, :2]  # (4,2)

                '''求解图像区域内的所有像素点坐标***************************************************************************************'''
                path = Path(img_corner.numpy())
                x, y = np.mgrid[:img_w, :img_h]
                points = np.vstack((x.ravel(), y.ravel())).T  # (HW,2) [x,y]
                mask = path.contains_points(points)
                path_points = points[np.where(mask)]  # (Nin,2) [x,y]
                img_inner_points = torch.from_numpy(path_points)  # (Nin,2) [x,y]
                if time_i == 0 and len(img_inner_points) <= 200: break
                if time_i != 0 and len(img_inner_points) <= 10: continue

                '''将2D区域内所有像素点project到3D LiDAR系下********************************************************************'''
                img_points_orisize = (img_inner_points - post_tran[:2].unsqueeze(0)).matmul(torch.inverse(post_rot.T[:2, :2]))  # (Nin,2)
                R = torch.inverse(lidar2img[:3, :3].T)
                T = lidar2img[:3, 3]

                fz = z_bottle + T[0] * R[0, 2] + T[1] * R[1, 2] + T[2] * R[2, 2]
                fm = img_points_orisize[:, 0] * R[0, 2] + img_points_orisize[:, 1] * R[1, 2] + R[2, 2]
                C = fz / fm  # (Nin)
                img_points_orisize_C = torch.cat([(img_points_orisize[:, 0] * C).unsqueeze(-1),
                                                  (img_points_orisize[:, 1] * C).unsqueeze(-1),
                                                  C.unsqueeze(-1)], dim=1)
                lidar_inner_points = (img_points_orisize_C - T.unsqueeze(0)).matmul(R)  # (Nin, 3)

                '''找到每个3D点在poster上的颜色索引,并put到原图上'''
                lidar_inner_points[:, :2] -= fake_3d_box[:, :2]
                lidar_inner_points = rotate_points_along_z(lidar_inner_points.unsqueeze(0), -1 * torch.Tensor([fake_box[0, 4]])).squeeze(0)  # (Nin,3)
                lidar_inner_points[:, 0] += l / 2.
                lidar_inner_points[:, 1] += w / 2.

                index_l = torch.clip(lidar_inner_points[:, 0] // delta_l, min=0, max=poster_l - 1).long()
                index_w = torch.clip((w - lidar_inner_points[:, 1]) // delta_w, min=0, max=poster_w - 1).long()
                selected_color = leaning_poster[use_poster_idx,:,index_w, index_l].T #(Nin, 3) bgr 0~1 gpu

                #contrast = round(random.uniform(0.7, 1.0), 10)
                #brightness = round(random.uniform(-0.3, 0.2), 10)
                #selected_color = selected_color * contrast + brightness
                #selected_color[selected_color > 1] = 1
                #selected_color[selected_color < 0] = 0

                selected_color = (selected_color-torch.from_numpy(mean).cuda()) / torch.from_numpy(std).cuda()#(Nin, 3) 归一化
                batch_inputs['img_inputs'][0][0][frame_i, (num_adj+1)*cam_i+time_i, :, img_inner_points[:,1], img_inner_points[:,0]] = selected_color.T

                if time_i == 0:
                    batch_inputs['gt_bboxes_3d'][0][frame_i].tensor = torch.cat([gt_box, fake_3d_box], 0)
                    gt_label = batch_inputs['gt_labels_3d'][0][frame_i]
                    batch_inputs['gt_labels_3d'][0][frame_i] = torch.cat([gt_label, torch.Tensor([0]).to(gt_label.device)])
                if only_apply_cur_frame and (time_i>2):#仅将补丁渲染至t=0的帧
                    break


            if use_next_poster:
                use_poster_idx += 1














