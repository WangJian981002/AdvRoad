
from __future__ import division

import os
import time
import warnings
from os import path as osp
import numpy as np
import cv2

import mmcv
import torchvision
import torch

import torch.nn.functional as F

import torch.distributed as dist

from mmcv.runner import get_dist_info, init_dist, load_checkpoint



from Spoofing3D.adv_utils.common_utils import rotate_points_along_z
from Spoofing3D.adv_utils.dcgan import DCGAN_G_CustomAspectRatio
from mmdet3d.apis import init_random_seed, train_model, init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed


try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)
import torch
from mmcv import Config

from Spoofing3D.adv_road.GAN_Spoofing import parse_args, put_poster_on_batch_inputs_eval, \
    put_poster_on_batch_inputs_eval_4D, mmlabDeNormalize, visulize_3dbox_to_cam, visulize_3dbox_to_cam_multiFrame
from mmdet3d.utils import setup_multi_processes


def inference(ind=None,path=None,is_4D=False):
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
    print(len(train_dataset))

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 64, 128
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(path))
    netG.eval()

    if 0:
        noise = torch.zeros(500, nosie_dim, 1, 1).normal_(0, 1)
        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear", align_corners=True)  # (m,3,200,300) 0~1 bgr
        image_poster = torch.flip(poster.clone(), [1])  # RGB
        torchvision.utils.save_image(image_poster, 'poster.jpg')
        assert False


    noise = torch.zeros(10, nosie_dim, 1, 1).normal_(0, 1)
    with torch.no_grad():
        poster = (netG(noise)+1)/2. #bgr
    poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear", align_corners=True)#(m,3,200,300) 0~1 bgr
    #torch.save(poster, 'poster.pth')
    image_poster = torch.flip(poster.clone(), [1])  # RGB
    torchvision.utils.save_image(image_poster, 'poster.jpg')
    poster = poster.to('cuda:0')


    #poster = torch.from_numpy((cv2.imread('Spoofing3D/init_poster.png')/255.0).astype(np.float32)).permute(2,0,1).unsqueeze(0).to('cuda:0')
    #poster = torch.load('poster.pth')[4].unsqueeze(0).to('cuda:0')

    if ind is None:
        ind = 777
    else:
        ind = int(ind)
    sigle_data_dict = train_dataset[ind]
    sample_info = train_dataset.data_infos[ind]

    #points = sigle_data_dict['points'][0].data.numpy()
    #points.tofile('points.bin')

    from mmcv.parallel import collate, scatter
    device = next(model.parameters()).device
    data = collate([sigle_data_dict], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device.index])[0]

    if not is_4D:
        ori_gt = data['gt_bboxes_3d'][0][0].tensor
        put_poster_on_batch_inputs_eval(poster, data)  # 将poster作用到img上
        #put_poster_on_batch_inputs_eval_cross_cam(poster, data)#, spoof_cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])  # 将poster作用到img上
        spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
    else:
        put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8)

    if 0: #only add points for BEVFusion
        default_lwh = (4., 1.8, 1.6)
        point_num = 20

        fake_gt = spoofed_gt[len(ori_gt):, :]
        print("spoof num = ", len(fake_gt))
        injected_point = []
        for si in range(len(fake_gt)):
            x_coord = torch.rand(point_num, 1) * default_lwh[0]*0.1 - default_lwh[0] / 2
            y_coord = torch.rand(point_num, 1) * default_lwh[1] - default_lwh[1] / 2
            z_coord = torch.rand(point_num, 1) * default_lwh[2] - default_lwh[2] / 2
            point_coord = torch.cat([x_coord, y_coord, z_coord], dim=1)
            point_coord = rotate_points_along_z(point_coord.unsqueeze(0), torch.Tensor([fake_gt[si,6]])).squeeze(0)
            point_coord = point_coord + fake_gt[si,:3].unsqueeze(0)
            point_coord[:,2] += default_lwh[2] / 2
            injected_point.append(point_coord.cuda())
        injected_point = torch.cat(injected_point, dim=0) #(m,3)
        cur_point = data['points'][0][0]
        injected_point = torch.cat([injected_point, cur_point.data[:len(injected_point),3:4], torch.zeros(len(injected_point),1).cuda()], dim=1)
        cur_point.data = torch.cat([cur_point.data, injected_point], dim=0)
        data['points'][0][0] = cur_point

    points = data['points'][0][0].data.cpu().numpy()
    points.tofile('points.bin')




    data['gt_bboxes_3d'][0][0].corners.numpy().tofile('anno.bin')
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    #print(result)

    predict_score = result[0]['pts_bbox']['scores_3d']
    mask = predict_score>0.15
    predicted_box = result[0]['pts_bbox']['boxes_3d']
    predicted_box.corners[mask].numpy().tofile('predict_anno.bin')


    if not is_4D:
        vis_imgs = []
        for i in range(6):
            img = data['img_inputs'][0][0][0][i].cpu()
            img = mmlabDeNormalize(img)  # rgb (H,W,3)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
            img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],sigle_data_dict['img_inputs'][0][5][i], 256, 704)
            #img = visulize_3dbox_to_cam(img, data['gt_bboxes_3d'][0][0].corners, sample_info, cams[i],sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i], 256,704)

            vis_imgs.append(img)

        vis_imgs = np.vstack(vis_imgs)
        cv2.imwrite('demo.png', vis_imgs)
    else:
        cam_total = []
        gt_coner = predicted_box.corners[mask]
        for cam_i in range(6):
            cam_col = []
            for adj_i in range(9):
                img = data['img_inputs'][0][0][0][cam_i * 9 + adj_i, :, :, :].cpu()
                img = mmlabDeNormalize(img)  # (3,H,W) rgb
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr

                img = visulize_3dbox_to_cam_multiFrame(img, gt_coner,
                                                       data['img_inputs'][0][1][0, adj_i * 6 + cam_i].cpu(),
                                                       data['img_inputs'][0][2][0, adj_i * 6 + cam_i].cpu(),
                                                       data['img_inputs'][0][3][0, adj_i * 6 + cam_i].cpu(),
                                                       data['img_inputs'][0][4][0, adj_i * 6 + cam_i].cpu(),
                                                       data['img_inputs'][0][5][0, adj_i * 6 + cam_i].cpu(),
                                                       256, 704)

                cam_col.append(img)
            cam_total.append(np.vstack(cam_col))
        cam_total = np.hstack(cam_total)
        cv2.imwrite('demo.png', cam_total)

def inference_whole(ind=None,path=None,is_4D=False):
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
    print(len(train_dataset))

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 64, 128
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(path))
    netG.eval()

    from tqdm import tqdm
    for ind in tqdm(range(len(train_dataset))):
        noise = torch.zeros(2, nosie_dim, 1, 1).normal_(0, 1)
        with torch.no_grad():
            poster = (netG(noise) + 1) / 2.  # bgr
        poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr
        poster = poster.to('cuda:0')

        sigle_data_dict = train_dataset[ind]
        sample_info = train_dataset.data_infos[ind]

        from mmcv.parallel import collate, scatter
        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        if not is_4D:
            ori_gt = data['gt_bboxes_3d'][0][0].tensor
            put_poster_on_batch_inputs_eval(poster, data)  # 将poster作用到img上
            #put_poster_on_batch_inputs_eval_cross_cam(poster, data)#, spoof_cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])  # 将poster作用到img上
            spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8)


        #data['gt_bboxes_3d'][0][0].corners.numpy().tofile('anno.bin')
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)


        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score>0.2
        predicted_box = result[0]['pts_bbox']['boxes_3d']
        #predicted_box.corners[mask].numpy().tofile('predict_anno.bin')


        if not is_4D:
            vis_imgs = []
            for i in range(6):
                img = data['img_inputs'][0][0][0][i].cpu()
                img = mmlabDeNormalize(img)  # rgb (H,W,3)
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],sigle_data_dict['img_inputs'][0][5][i], 256, 704)
                #img = visulize_3dbox_to_cam(img, data['gt_bboxes_3d'][0][0].corners, sample_info, cams[i],sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i], 256,704)

                vis_imgs.append(img)

            vis_imgs = np.vstack(vis_imgs)
            cv2.imwrite(os.path.join(args.work_dir, f"{ind:0>6}" + '.jpg'), vis_imgs)
        else:
            cam_total = []
            gt_coner = predicted_box.corners[mask]
            for cam_i in range(6):
                cam_col = []
                for adj_i in range(9):
                    img = data['img_inputs'][0][0][0][cam_i * 9 + adj_i, :, :, :].cpu()
                    img = mmlabDeNormalize(img)  # (3,H,W) rgb
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr

                    img = visulize_3dbox_to_cam_multiFrame(img, gt_coner,
                                                           data['img_inputs'][0][1][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][2][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][3][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][4][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][5][0, adj_i * 6 + cam_i].cpu(),
                                                           256, 704)

                    cam_col.append(img)
                cam_total.append(np.vstack(cam_col))
            cam_total = np.hstack(cam_total)
            cv2.imwrite(os.path.join(args.work_dir, f"{ind:0>6}" + '.jpg'), cam_total)

def inference_whole_poster(ind=None,path=None,is_4D=False):
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
    print(len(train_dataset))

    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    p_w, p_l = 64, 128
    poster_digital_size = [p_w, p_l]

    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.load_state_dict(torch.load(path))
    netG.eval()
    noise = torch.zeros(500, nosie_dim, 1, 1).normal_(0, 1)
    with torch.no_grad():
        poster = (netG(noise) + 1) / 2.  # bgr
    poster = F.interpolate(poster, size=poster_digital_size, mode="bilinear",align_corners=True)  # (m,3,200,300) 0~1 bgr
    image_poster = torch.flip(poster.clone(), [1])  # RGB
    torchvision.utils.save_image(image_poster, 'poster.jpg')
    torch.save(poster, 'GAN51-poster.pth')
    poster = poster.to('cuda:0')

    from tqdm import tqdm
    for pi in tqdm(range(poster.size(0))):
        if ind is None:
            ind = 1292
        else:
            ind = int(ind)

        sigle_data_dict = train_dataset[ind]
        sample_info = train_dataset.data_infos[ind]

        from mmcv.parallel import collate, scatter
        device = next(model.parameters()).device
        data = collate([sigle_data_dict], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device.index])[0]

        if not is_4D:
            ori_gt = data['gt_bboxes_3d'][0][0].tensor
            put_poster_on_batch_inputs_eval(poster, data,use_next_poster=False)  # 将poster作用到img上
            #put_poster_on_batch_inputs_eval_cross_cam(poster, data)#, spoof_cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])  # 将poster作用到img上
            spoofed_gt = data['gt_bboxes_3d'][0][0].tensor
        else:
            put_poster_on_batch_inputs_eval_4D(poster, data, num_adj=8,use_next_poster=False)


        #data['gt_bboxes_3d'][0][0].corners.numpy().tofile('anno.bin')
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)


        predict_score = result[0]['pts_bbox']['scores_3d']
        mask = predict_score>0.15
        predicted_box = result[0]['pts_bbox']['boxes_3d']
        #predicted_box.corners[mask].numpy().tofile('predict_anno.bin')


        if not is_4D:
            vis_imgs = []
            for i in range(6):
                img = data['img_inputs'][0][0][0][i].cpu()
                img = mmlabDeNormalize(img)  # rgb (H,W,3)
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img = visulize_3dbox_to_cam(img, predicted_box.corners[mask], sample_info, cams[i], sigle_data_dict['img_inputs'][0][4][i],sigle_data_dict['img_inputs'][0][5][i], 256, 704)
                #img = visulize_3dbox_to_cam(img, data['gt_bboxes_3d'][0][0].corners, sample_info, cams[i],sigle_data_dict['img_inputs'][0][4][i], sigle_data_dict['img_inputs'][0][5][i], 256,704)

                vis_imgs.append(img)

            vis_imgs = np.vstack(vis_imgs)
            cv2.imwrite(os.path.join(args.work_dir, 'demo' + str(pi) + '.jpg'), vis_imgs)
        else:
            cam_total = []
            gt_coner = predicted_box.corners[mask]
            for cam_i in range(6):
                cam_col = []
                for adj_i in range(9):
                    img = data['img_inputs'][0][0][0][cam_i * 9 + adj_i, :, :, :].cpu()
                    img = mmlabDeNormalize(img)  # (3,H,W) rgb
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr

                    img = visulize_3dbox_to_cam_multiFrame(img, gt_coner,
                                                           data['img_inputs'][0][1][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][2][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][3][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][4][0, adj_i * 6 + cam_i].cpu(),
                                                           data['img_inputs'][0][5][0, adj_i * 6 + cam_i].cpu(),
                                                           256, 704)

                    cam_col.append(img)
                cam_total.append(np.vstack(cam_col))
            cam_total = np.hstack(cam_total)
            cv2.imwrite(os.path.join(args.work_dir, 'demo' + str(pi) + '.jpg'), cam_total)
        poster = poster[1:, :, :, :]