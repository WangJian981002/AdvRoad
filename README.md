# Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving

This is a official code release of [AdvRoad](https://ieeexplore.ieee.org/abstract/document/10838314)（Physically Realizable Adversarial Creating Attack  Against Vision-Based BEV Space  3D Object Detection）. This code is mainly based on MMDetection3D（v1.0.0rc4）(https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc4/mmdet3d/models)、[BEVDet](https://github.com/HuangJunJie2017/BEVDet).




# Getting Started

The experimental environment and dependencies are consistent with those described in our [previous work](https://github.com/WangJian981002/BEVDet-Spoofing)

## Additional Installation

```
pip install lpips
```

Please download additional models and datasets from [here]( https://pan.baidu.com/s/1MaFBLg0Nj9h_LfDsJlnDSQ?pwd=dtv3  ), and placed in the specified folder, placed as follows

```
BEVDet-spoofing
├── mmdet3d
├── tools
├── configs
├── data
│   ├── Background_scene
├── Spoofing3D
├── work_dirs
│   ├── netG_epoch160.pth
│   ├── GAN44
│   ├── bevdet-r50
...
```



## Test

```
cd BEVDet-Spoofing

python Spoofing3D/adv_road/eval.py --config ./configs/bevdet/LidarSys-bevdet-r50-cbgs-spatial_0.6.py --checkpoint ./checkpoints/LidarSys_bevdet_r50_cbgs_spatial_06_mAP_3174_NDS_3939.pth --type two_stage --G_dir work_dirs/GAN44/netG_epoch16.pth
```

## Training

```
#pure Gan
python Spoofing3D/adv_road/train.py --config ./configs/bevdet/LidarSys-bevdet-r50-cbgs-spatial_0.6.py --checkpoint ./checkpoints/LidarSys_bevdet_r50_cbgs_spatial_06_mAP_3174_NDS_3939.pth --type pure
#adv train
python Spoofing3D/adv_road/train.py --config ./configs/bevdet/LidarSys-bevdet-r50-cbgs-spatial_0.6.py --checkpoint ./checkpoints/LidarSys_bevdet_r50_cbgs_spatial_06_mAP_3174_NDS_3939.pth --type train
```

## Inference

```
python Spoofing3D/adv_road/inference.py --config ./configs/bevdet/LidarSys-bevdet-r50-cbgs-spatial_0.6.py --checkpoint ./checkpoints/LidarSys_bevdet_r50_cbgs_spatial_06_mAP_3174_NDS_3939.pth --path your netG path --type whole(or inference or whole_poster) --ind 111
```


