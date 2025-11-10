from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp

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

from Spoofing3D.adv_road.GAN_Spoofing import parse_args, maskGT_put_poster_on_batch_inputs, \
    maskGT_put_poster_on_batch_inputs_4D
from Spoofing3D.adv_utils.common_utils import rotate_points_along_z
from Spoofing3D.adv_utils.dcgan import DCGAN_D_CustomAspectRatio, weights_init, DCGAN_G_CustomAspectRatio, SceneSet
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model, init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger, setup_multi_processes
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from matplotlib.path import Path

import lpips

def Training(is_4D=False):
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
    '''****************************************************************************************************************************************************************************************************************************************'''

    '''定义模型、读取参数文件、锁住参数和BN层****************************************'''
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
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    def fix_bn(m):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False

    model.apply(fix_bn)

    def act_bn(m):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True

    '''定义dataset和dataloader******************************************************'''
    train_dataset = build_dataset(cfg.data.train)
    logger.info(f'trainset contains {len(train_dataset)} frame')

    from mmdet.datasets import build_dataloader as build_mmdet_dataloader
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']
    train_loader = build_mmdet_dataloader(
        train_dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=cfg.data.get('persistent_workers', False))
    logger.info(f"trainloader contains {len(train_loader)} iter")

    '''超参数'''
    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    batch_size_D = 32
    scene_set_dir = 'data/Background_scene/RoadSnip-ori'

    lrD, lrG = 0.0001, 0.0001
    beta1 = 0.5
    max_epoch = 20
    G_iter_num = 4
    D_iter_num = 0  # 10
    clamp_lower, clamp_upper = -0.01, 0.01  # 限制discriminator参数的范围，参考WGAN

    p_w, p_l = 32, 64
    poster_digital_size = [p_w, p_l]
    adv_weight = 0.02
    tv_weight = 0  # 0.1
    D_weight = 1.
    print_iter = 1

    '''定义生成器，判别器、和场景数据集*************************************************'''
    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.apply(weights_init)
    if args.resume_netG != '':
        netG.load_state_dict(torch.load(args.resume_netG))
        logger.info(f"load G ckpt from: {str(args.resume_netG)} ")
    netG.to('cuda:0')
    netD = DCGAN_D_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ndf=64, n_extra_layers=0)
    netD.apply(weights_init)
    if args.resume_netD != '':
        netD.load_state_dict(torch.load(args.resume_netD))
        logger.info(f"load D ckpt from: {str(args.resume_netD)} ")
    netD.to('cuda:0')

    Scenedataset = SceneSet(scene_set_dir, imgsize=[poster_size_inside_net, poster_size_inside_net])
    Scenedataloader = DataLoader(Scenedataset, batch_size=batch_size_D, shuffle=True)
    logger.info(f"scene set contains {len(Scenedataset)} imgs")
    Scene_iter = iter(Scenedataloader)
    Scene_iter_count = 1

    '''定义优化器，并初始化*************************************************'''
    if args.adam:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
    else:
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=lrD)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lrG)
    one = torch.FloatTensor([1]).to('cuda:0')
    mone = (one * -1).to('cuda:0')

    errD_real, errD_fake = 0, 0
    '''训练poster*******************************************************************'''
    device = next(model.parameters()).device
    for epoch_i in range(max_epoch):
        # 开始一个epoch的训练
        for iter_i, batch_inputs in enumerate(train_loader):
            ############################
            # (1) Update D network
            ###########################
            if iter_i % G_iter_num == 0:
                for p in netD.parameters():
                    p.requires_grad = True
                netD.apply(act_bn)
                for p in netG.parameters():
                    p.requires_grad = False
                netG.apply(fix_bn)

                for _ in range(D_iter_num):
                    for p in netD.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)  # 限制discriminator参数的范围，参考WGAN

                    if Scene_iter_count < len(Scenedataloader):
                        realposter = Scene_iter.next()
                        Scene_iter_count += 1
                    else:
                        Scene_iter = iter(Scenedataloader)
                        realposter = Scene_iter.next()
                        Scene_iter_count = 2
                    bs_cur = realposter.size(0)
                    # realposter 0~1
                    netD.zero_grad()
                    errD_real = netD(realposter.to('cuda:0'))
                    errD_real.backward(one)

                    noise = torch.zeros(bs_cur, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
                    fakeposter = (netG(noise) + 1) / 2.0  # (0~1)
                    errD_fake = netD(fakeposter)
                    errD_fake.backward(mone)

                    optimizerD.step()
                    # logger.info(f"errD_real: {float(errD_real)}, errD_fake: {float(errD_fake)}")

            ############################
            # (2) Update G network
            ###########################
            for p in netG.parameters():
                p.requires_grad = True
            netG.apply(act_bn)
            for p in netD.parameters():
                p.requires_grad = False
            netD.apply(fix_bn)

            bs_cur = batch_inputs['img_inputs'][0].size(0) * 2  # 每帧最多放放在两个相机
            noise = torch.zeros(bs_cur, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
            GAN_out = (netG(noise) + 1) / 2.
            spoofing_poster = F.interpolate(GAN_out, size=poster_digital_size, mode="bilinear",
                                            align_corners=True)  # (m,3,200,300) 0~1 bgr

            batch_inputs = scatter(batch_inputs, [device.index])[0]  # 放到gpu上
            if not is_4D:
                maskGT_put_poster_on_batch_inputs(spoofing_poster, batch_inputs, is_bilinear=True,
                                                  mask_aug=False)  # 将poster作用到img上
            else:  # for bevdet4d
                maskGT_put_poster_on_batch_inputs_4D(spoofing_poster, batch_inputs, is_bilinear=True, num_adj=8,
                                                     use_next_poster=True)  # 将poster作用到img上

            '''
            vis_imgs = []
            frame_i = 2
            for i in range(6):
                img =batch_inputs['img_inputs'][0][frame_i][i].detach().cpu()
                img = mmlabDeNormalize(img)  # rgb (H,W,3)
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img = draw_box_from_batch(img,batch_inputs,frame_i,i)
                vis_imgs.append(img)

            vis_imgs = np.vstack(vis_imgs)
            cv2.imwrite('demo.png', vis_imgs)
            assert False
            '''

            '''
            cam_total = []
            frame_i = 6
            gt_coner = batch_inputs['gt_bboxes_3d'][frame_i].corners
            for cam_i in range(6):
                cam_col = []
                for adj_i in range(9):
                    img = batch_inputs['img_inputs'][0][frame_i][cam_i * 9 + adj_i, :, :, :].detach().cpu()
                    img = mmlabDeNormalize(img)  # (3,H,W) rgb
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # bgr
                    img = visulize_3dbox_to_cam_multiFrame(img, gt_coner,
                                                           batch_inputs['img_inputs'][1][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           batch_inputs['img_inputs'][2][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           batch_inputs['img_inputs'][3][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           batch_inputs['img_inputs'][4][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           batch_inputs['img_inputs'][5][frame_i,adj_i * 6 + cam_i].detach().cpu(),
                                                           256, 704)
                    cam_col.append(img)
                cam_total.append(np.hstack(cam_col))
            cam_total = np.vstack(cam_total)
            cv2.imwrite('demo.png', cam_total)
            assert False
            '''

            # continue_flag = 0
            # for gt_label in batch_inputs['gt_labels_3d']:
            #    if len(gt_label) == 0: continue_flag = 1
            # if continue_flag: continue

            output = model(return_loss=True, **batch_inputs)  # 前向传播计算loss
            adv_loss = 0
            for k in output.keys():
                if 'heatmap' in k:
                    adv_loss += output[k]
                else:
                    adv_loss += 1 * output[k]  # 为了节省显存
                # adv_loss += output[k]

            tv_loss = torch.sqrt(
                1e-7 + torch.sum((spoofing_poster[:, :, :p_w - 1, :p_l - 1] - spoofing_poster[:, :, 1:, :p_l - 1]) ** 2,
                                 dim=1) +
                torch.sum((spoofing_poster[:, :, :p_w - 1, :p_l - 1] - spoofing_poster[:, :, :p_w - 1, 1:]) ** 2,
                          dim=1)).mean()

            D_loss = netD(GAN_out)

            # loss = adv_weight * adv_loss + tv_weight * tv_loss + D_weight* D_loss
            loss = adv_weight * adv_loss

            loss.backward()
            if iter_i % G_iter_num == 0:
                optimizerG.step()
                optimizerG.zero_grad()

            if iter_i % G_iter_num == 0:
                logger.info(
                    f"epoch: {epoch_i}, iter: {iter_i}, adv_loss: {float(adv_loss)}, D_loss: {float(D_loss)}, tv_loss: {float(tv_loss)}, errD_real: {float(errD_real)}, errD_fake: {float(errD_fake)}")
            if iter_i % 200 == 0:
                vis_img = torch.flip(GAN_out.cpu().detach(), [1])  # (m,3,256,256) 0~1 rgb
                torchvision.utils.save_image(vis_img, os.path.join(cfg.work_dir, f"e{epoch_i + 1}i{iter_i}.jpg"))

            # torch.cuda.empty_cache()

        # 保存海报
        torch.save(netG.state_dict(), os.path.join(cfg.work_dir, f'netG_epoch{epoch_i + 1}.pth'))
        torch.save(netD.state_dict(), os.path.join(cfg.work_dir, f'netD_epoch{epoch_i + 1}.pth'))

def Training_pure_GAN():
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
    '''****************************************************************************************************************************************************************************************************************************************'''


    def fix_bn(m):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False

    def act_bn(m):
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True


    '''超参数'''
    poster_size_inside_net = 32
    row_size = [poster_size_inside_net * 2, poster_size_inside_net * 2]
    nosie_dim = 100
    batch_size_D = 32

    lrD, lrG = 0.0001, 0.00003
    beta1 = 0.5
    max_epoch = 500
    G_iter_num = 1
    D_iter_num = 20
    clamp_lower, clamp_upper = -0.01, 0.01  # 限制discriminator参数的范围，参考WGAN


    print_iter = 1

    '''定义生成器，判别器、和场景数据集*************************************************'''
    netG = DCGAN_G_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ngf=64, n_extra_layers=0)
    netG.apply(weights_init)
    netG.to('cuda:0')
    netD = DCGAN_D_CustomAspectRatio(row_size, nz=nosie_dim, nc=3, ndf=64, n_extra_layers=0)
    netD.apply(weights_init)
    netD.to('cuda:0')

    Scenedataset = SceneSet('data/Background_scene/RoadSnip-ori', imgsize=[poster_size_inside_net, poster_size_inside_net])
    Scenedataloader = DataLoader(Scenedataset, batch_size=batch_size_D, shuffle=True)
    logger.info(f"scene set contains {len(Scenedataset)} imgs")
    Scene_iter = iter(Scenedataloader)
    Scene_iter_count = 1

    '''定义优化器，并初始化*************************************************'''
    if args.adam:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
    else:
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=lrD)
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lrG)
    one = torch.FloatTensor([1]).to('cuda:0')
    mone = (one * -1).to('cuda:0')

    '''训练poster*******************************************************************'''
    for epoch_i in range(max_epoch):
        # 开始一个epoch的训练
        for iter_i in range(1000):
            if iter_i % G_iter_num == 0:
                ############################
                # (1) Update D network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = True
                netD.apply(act_bn)
                for p in netG.parameters():
                    p.requires_grad = False
                netG.apply(fix_bn)

                for _ in range(D_iter_num):
                    for p in netD.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)  # 限制discriminator参数的范围，参考WGAN

                    if Scene_iter_count < len(Scenedataloader):
                        realposter = Scene_iter.next()
                        Scene_iter_count += 1
                    else:
                        Scene_iter = iter(Scenedataloader)
                        realposter = Scene_iter.next()
                        Scene_iter_count = 2
                    bs_cur = realposter.size(0)
                    # realposter 0~1
                    netD.zero_grad()
                    errD_real = netD(realposter.to('cuda:0'))
                    errD_real.backward(one)

                    #noise = torch.zeros(bs_cur, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
                    noise = torch.cuda.FloatTensor(bs_cur, nosie_dim, 1, 1).normal_(0, 1)
                    fakeposter = (netG(noise) + 1) / 2.0  # (0~1)
                    errD_fake = netD(fakeposter)
                    errD_fake.backward(mone)
                    optimizerD.step()



            ############################
            # (2) Update G network
            ###########################
            for p in netG.parameters():
                p.requires_grad = True
            netG.apply(act_bn)
            for p in netD.parameters():
                p.requires_grad = False
            netD.apply(fix_bn)

            bs_cur = 20
            #noise = torch.zeros(bs_cur, nosie_dim, 1, 1).normal_(0, 1).to('cuda:0')
            noise = torch.cuda.FloatTensor(bs_cur, nosie_dim, 1, 1).normal_(0, 1)
            GAN_out = (netG(noise) + 1) / 2.

            D_loss = netD(GAN_out)


            D_loss.backward()
            if iter_i % G_iter_num == 0:
                optimizerG.step()
                optimizerG.zero_grad()

            if iter_i % G_iter_num == 0:
                logger.info(f"epoch: {epoch_i}, iter: {iter_i}, errD_real: {float(errD_real)}, errD_fake: {float(errD_fake)}, errG: {float(D_loss)}")
            if iter_i % 1000 == 0:
                vis_img = torch.flip(GAN_out.cpu().detach(), [1])  # (m,3,256,256) 0~1 rgb
                torchvision.utils.save_image(vis_img, os.path.join(cfg.work_dir, f"e{epoch_i + 1}i{iter_i}.jpg"))

            torch.cuda.empty_cache()

        # 保存海报
        if (epoch_i + 1) % 10 == 0:
            torch.save(netG.state_dict(), os.path.join(cfg.work_dir, f'netG_epoch{epoch_i + 1}.pth'))
            torch.save(netD.state_dict(), os.path.join(cfg.work_dir, f'netD_epoch{epoch_i + 1}.pth'))