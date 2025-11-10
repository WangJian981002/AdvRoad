import copy
import warnings
import time

import cv2


import mmcv
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmcv.ops import box_iou_rotated

from Spoofing3D.adv_road.GAN_Spoofing import put_poster_on_batch_inputs_eval, put_poster_on_batch_inputs_eval_4D, \
    parse_args, mmlabDeNormalize, maskGT_put_poster_on_batch_inputs_v2, maskGT_put_poster_on_batch_inputs_4D
from Spoofing3D.adv_utils.common_utils import rotate_points_along_z
from Spoofing3D.adv_utils.dcgan import DCGAN_G_CustomAspectRatio
from mmdet3d.apis import init_random_seed, train_model, init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

import lpips

try:

    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)
def eval(G_dir,score_thr=0.1, iou_thr=[0.1,0.3,0.5,0.7], center_thr=[0.5,1,2,3], is_4D=False, is_add_point=False):
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs/debug')
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    from mmcv.parallel import collate, scatter
    #model = init_model(args.config, args.checkpoint, device='cuda:0')
    #print(model)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = cfg.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()


    datasets = [build_dataset(cfg.data.val)]
    train_dataset = datasets[0]
    print(f"测试集共包含{len(train_dataset)}帧")

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 32, 64
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(G_dir))
    netG.to('cuda:0')
    netG.eval()



    total_test_frame = 1000
    valid_spoof = 0
    success_spoof_iou = {}
    success_spoof_centerDistant = {}
    for thr in iou_thr:
        success_spoof_iou['iou_%s'%str(thr)] = 0
    for thr in center_thr:
        success_spoof_centerDistant["center_%s"%str(thr)] = 0


    inds = np.random.choice(list(range(len(train_dataset))), total_test_frame, replace=False)
    for jj,ind in enumerate(inds):
        sigle_data_dict = train_dataset[ind]
        #sample_info = train_dataset.data_infos[ind]

        noise = torch.zeros(2, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr
        #poster = torch.rand(poster.size()).to('cuda:0')
        #poster = torch.from_numpy((cv2.imread('Spoofing3D/init_poster.png') / 255.0).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to('cuda:0')

        #poster_collection = torch.load('BF-SwinT-rgb-selected.pth')
        #poster_collection = torch.flip(poster_collection, [1])
        #poster = poster_collection[np.random.randint(poster_collection.size(0))].unsqueeze(0).to('cuda:0')


        #poster = torch.load('Spoofing3D/work_dir/try24/poster_8.pth').to('cuda:0')
        #poster = poster.permute(2,0,1).unsqueeze(0)


        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        ori_gt = data['gt_bboxes_3d'][0][0].tensor

        if not is_4D:
            put_poster_on_batch_inputs_eval(poster, data, mask_aug=False,use_next_poster=False)  # 将poster作用到img上
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8,use_next_poster=False)  # 将poster作用到img上

        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        if len(ori_gt) == len(spoofed_gt): continue
        fake_gt = spoofed_gt[len(ori_gt):,:]
        valid_spoof += len(fake_gt)

        if is_add_point and len(fake_gt)>0 :  # only add points for BEVFusion
            default_lwh = (4., 1.8, 1.6)
            point_num = 20

            injected_point = []
            for si in range(len(fake_gt)):
                x_coord = torch.rand(point_num, 1) * default_lwh[0] * 0.1 - default_lwh[0]*0.1 / 2
                y_coord = torch.rand(point_num, 1) * default_lwh[1] - default_lwh[1] / 2
                z_coord = torch.rand(point_num, 1) * default_lwh[2] - default_lwh[2] / 2
                point_coord = torch.cat([x_coord, y_coord, z_coord], dim=1)
                point_coord = rotate_points_along_z(point_coord.unsqueeze(0), torch.Tensor([fake_gt[si, 6]])).squeeze(0)
                point_coord = point_coord + fake_gt[si, :3].unsqueeze(0)
                point_coord[:, 2] += default_lwh[2] / 2
                injected_point.append(point_coord.cuda())
            injected_point = torch.cat(injected_point, dim=0)  # (m,3)
            cur_point = data['points'][0][0]
            injected_point = torch.cat(
                [injected_point, cur_point.data[:len(injected_point), 3:4], torch.zeros(len(injected_point), 1).cuda()],
                dim=1)
            cur_point.data = torch.cat([cur_point.data, injected_point], dim=0)
            data['points'][0][0] = cur_point



        with torch.no_grad(): #前向推理
            result = model(return_loss=False, rescale=True, **data)

        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score > score_thr
        predicted_box = result[0]['pts_bbox']['boxes_3d'].tensor[mask] #(N,9)
        if len(predicted_box) ==0 : continue

        fake_bev = fake_gt[:, [0, 1, 3, 4, 6]]
        pred_bev = predicted_box[:, [0, 1, 3, 4, 6]]
        bev_iou = box_iou_rotated(fake_bev, pred_bev)

        for thr in iou_thr:
            success_spoof_iou['iou_%s' % str(thr)] += (torch.max(bev_iou,dim=1)[0] > thr).sum()


        for i in range(len(fake_bev)):
            min_dis = torch.sqrt(torch.sum((pred_bev[:,:2] - fake_bev[i][:2].unsqueeze(0))**2, dim=1)).min()
            for thr in center_thr:
                if min_dis <thr: success_spoof_centerDistant["center_%s"%str(thr)] += 1

        print("\r", f"{jj}/{total_test_frame}", f"有效伪造目标个数{valid_spoof}", f"iou_thr=0.5，成功伪造{success_spoof_iou['iou_%s' % str(0.5)]}个，成功率为{success_spoof_iou['iou_%s' % str(0.5)]*1./valid_spoof}",
              f"center dis=1，成功伪造{success_spoof_centerDistant['center_%s'%str(1)]}个，成功率为{success_spoof_centerDistant['center_%s'%str(1)]*1./valid_spoof}",end="")

        if jj % 100 == 0:
            print(f"有效伪造目标个数{valid_spoof}")
            for thr in iou_thr:
                print(f"BEV iou阈值为{thr}时，成功伪造{success_spoof_iou['iou_%s' % str(thr)]}个，成功率为{success_spoof_iou['iou_%s' % str(thr)] * 1. / valid_spoof}")
            for thr in center_thr:
                print(f"中心点距离阈值为{thr}时，成功伪造{success_spoof_centerDistant['center_%s' % str(thr)]}个，成功率为{success_spoof_centerDistant['center_%s' % str(thr)] * 1. / valid_spoof}")

    print("Finished")
    print(f"有效伪造目标个数{valid_spoof}")

    for thr in iou_thr:
        print(f"BEV iou阈值为{thr}时，成功伪造{success_spoof_iou['iou_%s' % str(thr)]}个，成功率为{success_spoof_iou['iou_%s' % str(thr)] * 1. / valid_spoof}")
    for thr in center_thr:
        print(f"中心点距离阈值为{thr}时，成功伪造{success_spoof_centerDistant['center_%s'%str(thr)]}个，成功率为{success_spoof_centerDistant['center_%s'%str(thr)] * 1. / valid_spoof}")

def eval_LPIPS(G_dir, is_4D=False):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs/debug')
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    from mmcv.parallel import collate, scatter
    #model = init_model(args.config, args.checkpoint, device='cuda:0')
    #print(model)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = cfg.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()


    datasets = [build_dataset(cfg.data.val)]
    train_dataset = datasets[0]
    print(f"测试集共包含{len(train_dataset)}帧")

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 32, 64
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(G_dir))
    netG.to('cuda:0')
    netG.eval()

    lpips_model = lpips.LPIPS(net="alex")

    total_test_frame = 1000
    lpips_total = 0
    valid_cal = 0



    inds = np.random.choice(list(range(len(train_dataset))), total_test_frame, replace=False)
    #inds = [2937]
    for jj,ind in enumerate(inds):
        sigle_data_dict = train_dataset[ind]
        #sample_info = train_dataset.data_infos[ind]

        if not is_4D:
            #scene_image = mmlabDeNormalize(sigle_data_dict['img_inputs'][0][0][1,:,:,:]) #rgb (H,W,3) 0~255 numpy
            #cv2.cvtColor(scene_image, cv2.COLOR_RGB2BGR, scene_image)
            #cv2.imwrite('demo0.png', scene_image)

            ori_image = mmlabDeNormalize(sigle_data_dict['img_inputs'][0][0][1,:,:,224:480]) #rgb (H,W,3) 0~255 numpy
            #vis_ori = copy.deepcopy(ori_image)
            #cv2.cvtColor(vis_ori, cv2.COLOR_RGB2BGR, vis_ori)
            #cv2.imwrite('demo1.png', vis_ori)
            ori_image = (torch.from_numpy(ori_image).permute(2,0,1)/255.).unsqueeze(0).float() #rgb (1,3,h,w) 0~1
        else:
            ori_image = mmlabDeNormalize(sigle_data_dict['img_inputs'][0][0][1*9+0, :, :, 224:480])  # rgb (H,W,3) 0~255 numpy
            #cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR, ori_image)
            #cv2.imwrite('demo1.png', ori_image)
            ori_image = (torch.from_numpy(ori_image).permute(2, 0, 1) / 255.).unsqueeze(0).float()  # rgb (1,3,h,w) 0~1

        noise = torch.zeros(2, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr
        #poster = torch.rand(poster.size()).to('cuda:0')
        #poster = torch.from_numpy((cv2.imread('Spoofing3D/init_poster.png') / 255.0).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to('cuda:0')

        #poster_collection = torch.load('BF-SwinT-rgb-selected.pth')
        #poster_collection = torch.flip(poster_collection, [1])
        #poster = poster_collection[np.random.randint(poster_collection.size(0))].unsqueeze(0).to('cuda:0')

        #poster = torch.load('Spoofing3D/work_dir/try24/poster_8.pth').to('cuda:0')
        #poster = poster.permute(2, 0, 1).unsqueeze(0)


        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        ori_gt = data['gt_bboxes_3d'][0][0].tensor

        if not is_4D:
            put_poster_on_batch_inputs_eval(poster, data, mask_aug=False,use_next_poster=False,spoof_cams=['CAM_FRONT'])  # 将poster作用到img上
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8,use_next_poster=False,spoof_cams=['CAM_FRONT'])  # 将poster作用到img上

        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        if len(ori_gt) == len(spoofed_gt): continue

        valid_cal+=1
        if not is_4D:
            spoof_image = data['img_inputs'][0][0][0][1].cpu()
            spoof_image = mmlabDeNormalize(spoof_image[:, :, 224:480])  # rgb (H,W,3) 0~255 numpy
            #vis_spoof = copy.deepcopy(spoof_image)
            #cv2.cvtColor(vis_spoof, cv2.COLOR_RGB2BGR, vis_spoof )
            #cv2.imwrite('demo2.png', vis_spoof)
            #assert False
            spoof_image = (torch.from_numpy(spoof_image).permute(2, 0, 1) / 255.).unsqueeze(0).float()  # rgb (1,3,h,w) 0~1
        else:
            spoof_image = data['img_inputs'][0][0][0][1*9+0].cpu()
            spoof_image = mmlabDeNormalize(spoof_image[:, :, 224:480])  # rgb (H,W,3) 0~255 numpy
            #cv2.cvtColor(spoof_image, cv2.COLOR_RGB2BGR, spoof_image )
            #cv2.imwrite('demo2.png', spoof_image)
            spoof_image = (torch.from_numpy(spoof_image).permute(2, 0, 1) / 255.).unsqueeze(0).float()  # rgb (1,3,h,w) 0~1

        lp_distance = lpips_model(ori_image, spoof_image)
        lpips_total += lp_distance




        print("\r",  f"{jj}/{total_test_frame}", f"valid spoof = {valid_cal}", f"Avg. LPIPS= {lpips_total/valid_cal}" ,end="")
    print(f"Avg. LPIPS= {lpips_total / valid_cal}")


def eval_two_stage(G_dir, score_thr=0.1, iou_thr=[0.1, 0.3, 0.5, 0.7], center_thr=[0.5, 1, 2, 3], is_4D=False,
                   is_add_point=False):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs/debug')
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    from mmcv.parallel import collate, scatter
    # model = init_model(args.config, args.checkpoint, device='cuda:0')
    # print(model)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = cfg.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = cfg
    model.to('cuda:0')
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    datasets = [build_dataset(cfg.data.val)]
    eval_dataset = datasets[0]
    print(f"测试集共包含{len(eval_dataset)}帧")

    train_dataset = build_dataset(cfg.data.test)
    assert len(eval_dataset) == len(train_dataset)

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 32, 64
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(G_dir))
    netG.to('cuda:0')
    netG.eval()
    for p in netG.parameters():
        p.requires_grad = False

    total_test_frame = 1000  # 1000
    valid_spoof = 0
    success_spoof_iou = {}
    success_spoof_centerDistant = {}
    for thr in iou_thr:
        success_spoof_iou['iou_%s' % str(thr)] = 0
    for thr in center_thr:
        success_spoof_centerDistant["center_%s" % str(thr)] = 0

    inds = np.random.choice(list(range(len(eval_dataset))), total_test_frame, replace=False)

    # noise = torch.zeros(1, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
    # noise = nn.Parameter(noise, requires_grad=True)
    # optimizer_noise = torch.optim.Adam([noise], lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    for jj, ind in enumerate(inds):
        # ind = 2155
        sigle_data_dict = train_dataset[ind]

        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        '''更新noise'''
        noise = torch.zeros(1, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
        noise = nn.Parameter(noise, requires_grad=True)
        optimizer_noise = torch.optim.Adam([noise], lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

        for noise_update_iter in range(30):
            poster = (netG(noise) + 1) / 2.  # bgr
            poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",
                                   align_corners=True)  # (1,3,200,300) 0~1 bgr

            input_data = copy.deepcopy(data)
            if not is_4D:
                if noise_update_iter == 0:
                    location_dict = maskGT_put_poster_on_batch_inputs_v2(poster, input_data, mask_aug=False,
                                                                         location_dict=None,
                                                                         use_next_poster=False)
                    location_dict = None
                else:
                    empty_ = maskGT_put_poster_on_batch_inputs_v2(poster, input_data, mask_aug=False,
                                                                  location_dict=copy.deepcopy(location_dict),
                                                                  use_next_poster=False)
            else:
                maskGT_put_poster_on_batch_inputs_4D(poster, input_data, is_bilinear=True, num_adj=8,
                                                     use_next_poster=False)

            '''
            vis_imgs = []
            frame_i = 0
            for i in range(6):
                img = input_data['img_inputs'][0][frame_i][i].detach().cpu()
                img = mmlabDeNormalize(img)  # rgb (H,W,3)
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img = draw_box_from_batch(img, input_data, frame_i, i)
                vis_imgs.append(img)

            vis_imgs = np.vstack(vis_imgs)
            cv2.imwrite('demo1.png', vis_imgs)
            assert False
            '''

            if len(input_data['gt_bboxes_3d'][0].tensor) == 0:
                break
            output = model(return_loss=True, **input_data)  # 前向传播计算loss
            adv_loss = 0
            for k in output.keys():
                adv_loss += output[k]
            # print(noise_update_iter, adv_loss)
            optimizer_noise.zero_grad()
            adv_loss.backward()
            optimizer_noise.step()

        '''eval'''
        sigle_data_dict = eval_dataset[ind]
        sample_info = eval_dataset.data_infos[ind]

        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",
                               align_corners=True)  # (m,3,200,300) 0~1 bgr

        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        ori_gt = data['gt_bboxes_3d'][0][0].tensor

        if not is_4D:
            put_poster_on_batch_inputs_eval(poster, data, mask_aug=False,
                                            use_next_poster=False)  # 将poster作用到img上
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8, use_next_poster=False,
                                               only_apply_cur_frame=False)  # 将poster作用到img上

        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        if len(ori_gt) == len(spoofed_gt): continue
        fake_gt = spoofed_gt[len(ori_gt):, :]
        valid_spoof += len(fake_gt)

        if is_add_point and len(fake_gt) > 0:  # only add points for BEVFusion
            default_lwh = (4., 1.8, 1.6)
            point_num = 20

            injected_point = []
            for si in range(len(fake_gt)):
                x_coord = torch.rand(point_num, 1) * default_lwh[0] * 0.1 - default_lwh[0] * 0.1 / 2
                y_coord = torch.rand(point_num, 1) * default_lwh[1] - default_lwh[1] / 2
                z_coord = torch.rand(point_num, 1) * default_lwh[2] - default_lwh[2] / 2
                point_coord = torch.cat([x_coord, y_coord, z_coord], dim=1)
                point_coord = rotate_points_along_z(point_coord.unsqueeze(0),
                                                    torch.Tensor([fake_gt[si, 6]])).squeeze(0)
                point_coord = point_coord + fake_gt[si, :3].unsqueeze(0)
                point_coord[:, 2] += default_lwh[2] / 2
                injected_point.append(point_coord.cuda())
            injected_point = torch.cat(injected_point, dim=0)  # (m,3)
            cur_point = data['points'][0][0]
            injected_point = torch.cat(
                [injected_point, cur_point.data[:len(injected_point), 3:4],
                 torch.zeros(len(injected_point), 1).cuda()],
                dim=1)
            cur_point.data = torch.cat([cur_point.data, injected_point], dim=0)
            data['points'][0][0] = cur_point

        with torch.no_grad():  # 前向推理
            result = model(return_loss=False, rescale=True, **data)

        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score > score_thr
        predicted_box = result[0]['pts_bbox']['boxes_3d'].tensor[mask]  # (N,9)
        if len(predicted_box) == 0: continue

        '''
        vis_imgs = []
        for i in range(6):
            img = data['img_inputs'][0][0][0][i].cpu()
            img = mmlabDeNormalize(img)  # rgb (H,W,3)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            img = visulize_3dbox_to_cam(img, result[0]['pts_bbox']['boxes_3d'].corners[mask], sample_info, cams[i],
                                        sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i],
                                        256, 704)
            # img = visulize_3dbox_to_cam(img, data['gt_bboxes_3d'][0][0].corners, sample_info, cams[i],sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i], 256,704)

            vis_imgs.append(img)

        vis_imgs = np.vstack(vis_imgs)
        cv2.imwrite('demo2.png', vis_imgs)
        assert False
        '''

        fake_bev = fake_gt[:, [0, 1, 3, 4, 6]]
        pred_bev = predicted_box[:, [0, 1, 3, 4, 6]]
        bev_iou = box_iou_rotated(fake_bev, pred_bev)

        for thr in iou_thr:
            success_spoof_iou['iou_%s' % str(thr)] += (torch.max(bev_iou, dim=1)[0] > thr).sum()

        for i in range(len(fake_bev)):
            min_dis = torch.sqrt(torch.sum((pred_bev[:, :2] - fake_bev[i][:2].unsqueeze(0)) ** 2, dim=1)).min()
            for thr in center_thr:
                if min_dis < thr: success_spoof_centerDistant["center_%s" % str(thr)] += 1

        print("\r", f"{jj}/{total_test_frame}", f"有效伪造目标个数{valid_spoof}",
              f"iou_thr=0.5，成功伪造{success_spoof_iou['iou_%s' % str(0.5)]}个，成功率为{success_spoof_iou['iou_%s' % str(0.5)] * 1. / valid_spoof}",
              f"center dis=1，成功伪造{success_spoof_centerDistant['center_%s' % str(1)]}个，成功率为{success_spoof_centerDistant['center_%s' % str(1)] * 1. / valid_spoof}",
              end="")

        if jj % 100 == 0:
            print(f"有效伪造目标个数{valid_spoof}")
            for thr in iou_thr:
                print(
                    f"BEV iou阈值为{thr}时，成功伪造{success_spoof_iou['iou_%s' % str(thr)]}个，成功率为{success_spoof_iou['iou_%s' % str(thr)] * 1. / valid_spoof}")
            for thr in center_thr:
                print(
                    f"中心点距离阈值为{thr}时，成功伪造{success_spoof_centerDistant['center_%s' % str(thr)]}个，成功率为{success_spoof_centerDistant['center_%s' % str(thr)] * 1. / valid_spoof}")

    print("Finished")
    print(f"有效伪造目标个数{valid_spoof}")

    for thr in iou_thr:
        print(
            f"BEV iou阈值为{thr}时，成功伪造{success_spoof_iou['iou_%s' % str(thr)]}个，成功率为{success_spoof_iou['iou_%s' % str(thr)] * 1. / valid_spoof}")
    for thr in center_thr:
        print(
            f"中心点距离阈值为{thr}时，成功伪造{success_spoof_centerDistant['center_%s' % str(thr)]}个，成功率为{success_spoof_centerDistant['center_%s' % str(thr)] * 1. / valid_spoof}")