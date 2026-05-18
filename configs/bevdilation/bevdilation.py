_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
]
'''

'''
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-54.0, -54.0, -3.0, 54.0, 54.0, 5.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 5,
    'input_size': (448, 800),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.44),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'random_crop_height': True,
    'vflip':True,
    'resize_test': 0.04,

    'pmd': dict(
        brightness_delta=32,
        contrast_lower=0.5,
        contrast_upper=1.5,
        saturation_lower=0.5,
        saturation_upper=1.5,
        hue_delta=18,
        rate=0.5
    )
}

grid_config = {
    'x': [-54.0, 54.0, 0.6],
    'y': [-54.0, 54.0, 0.6],
    'z': [-3, 5, 8],
    'depth': [1.0, 60.0, 0.5],
}


# Model
voxel_size = [0.075, 0.075, 0.2]

feat_bev_img_dim = 32
img_feat_dim = 256
model = dict(
    type='BEVDilation',
    use_grid_mask=True,
    # camera
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512, 1024, 2048],
        out_channels=img_feat_dim,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=img_feat_dim,
        out_channels=feat_bev_img_dim,
        downsample=8,
        with_depth_from_lidar=True),

    # lidar
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(120000, 160000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        base_channels=24,
        sparse_shape=[41, 1440, 1440],
        output_channels=192,
        order=('conv', 'norm', 'act'),
        encoder_channels=((24, 24, 48),
                          (48, 48, 96),
                          (96, 96, 192),
                          (192, 192)),
        encoder_paddings=((0, 0, 1),
                          (0, 0, 1),
                          (0, 0, [0, 1, 1]),
                          (0, 0)),
        block_type='basicblock',
        vx_config=dict(
            fg_thr=0.4,
            curve_rank=8,
            curve_template_path='./data/hilbert/curve_template_2d_rank_8.pth',
            img_channels=32,
            lidar_channels=192,
            img_encoder_num=[2, 1, 1],
            img_downstrides=[1, 2, 2],
            bev_shape=(180, 180),
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            downstride=8,
        )
        ),

    # pts_backbone=dict(
    #     type='SECOND',
    #     in_channels=384,
    #     out_channels=[192, 384],
    #     layer_nums=[8, 8],
    #     layer_strides=[1, 2],
    #     norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    #     conv_cfg=dict(type='Conv2d', bias=False)),
    # pts_neck=dict(
    #     type='SECONDFPN',
    #     in_channels=[192, 384],
    #     out_channels=[256, 256],
    #     upsample_strides=[1, 2],
    #     norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    #     upsample_cfg=dict(type='deconv', bias=False),
    #     use_conv_for_no_stride=True),
    pts_backbone=dict(
        type='Cascade_SBDB',
        num_layers=4,
        num_SBB=[2, 1, 1],
        down_strides=[1, 2, 2],
        feature_dim=256,
        input_channels=192,
        groups=16,
        groups_img=2),

    # head
    pts_bbox_head=dict(
        type='DALHead',

        # DAL
        feat_bev_img_dim=feat_bev_img_dim,
        img_feat_dim=img_feat_dim,
        sparse_fuse_layers=2,
        dense_fuse_layers=2,
        instance_attn=False,

        # Transfusion
        num_proposals=200,
        in_channels=256,
        hidden_channel=256,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        auxiliary=True,
        common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2],
            vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_bevmask=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0)
        ),
    train_cfg=dict(pts=dict(
        dataset='nuScenes',
        point_cloud_range=point_cloud_range,
        grid_size=[1440, 1440, 40],
        voxel_size=voxel_size,
        out_size_factor=8,
        gaussian_overlap=0.1,
        min_radius=2,
        pos_weight=-1,
        code_weights=[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ],
        assigner=dict(
            type='HungarianAssigner3D',
            iou_calculator=dict(
                type='BboxOverlaps3D', coordinate='lidar'),
            cls_cost=dict(
                type='FocalLossCost',
                gamma=2.0,
                alpha=0.25,
                weight=0.15),
            reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
            iou_cost=dict(type='IoU3DCost', weight=0.25)))),
    test_cfg=dict(pts=dict(
        dataset='nuScenes',
        grid_size=[1440, 1440, 40],
        img_feat_downsample=8,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        pc_range=point_cloud_range[:2],
        nms_type=None)),
)

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5 * 2, 22.5 * 2),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    tran_lim=[0.5, 0.5, 0.5]
)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'bevdetv3-nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True, opencv_pp=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='ToEgo'),
    dict(type='LoadAnnotations'),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='VelocityAug'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config=grid_config),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
                                 'img_inputs', 'gt_depth',
                                 'gt_bboxes_ignore'
                                 ])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=False, opencv_pp=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='ToEgo'),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_depth'])
        ])
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=3,  # Step 4: matches paper's per-GPU batch; original was 4 (16x A6000)
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            img_info_prototype='bevdet',
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(pipeline=test_pipeline, classes=class_names,
             modality=input_modality,
             ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
             img_info_prototype='bevdet'),
    test=dict(pipeline=test_pipeline, classes=class_names,
              modality=input_modality,
              ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
              img_info_prototype='bevdet'))

evaluation = dict(interval=20, pipeline=test_pipeline)
# Override default_runtime.py's epoch-based dict(interval=1) so a Ctrl-C at
# any iter leaves behind a usable checkpoint. max_keep_ckpts bounds disk.
checkpoint_config = dict(interval=2000, by_epoch=False, max_keep_ckpts=5)
# optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01, paramwise_cfg=dict(
    custom_keys={'img_backbone': dict(lr_mult=0.01),}))  # for 64 total batch size

# A30 (24GB) memory fit: at samples_per_gpu=3 we expect 18-22 GB/GPU (still
# under 24 GB available, confirmed by the Step 4 memory test), so fp16 isn't
# needed — and was actively harmful when we tried it (BCE overflow +
# dynamic-scaler skip stalled learning).
# 3 samples/GPU * 4 GPUs * 1 cumulative_iters = 12 effective batch
# (vs paper's 24 with 8 GPUs × 3 × 1).
#
# Use NanSkipGradientCumulativeOptimizerHook (defined in
# mmdet3d/core/hook/nan_skip_optimizer.py) so that a single bad-batch
# nan/inf in the loss or gradients is dropped on the floor instead of
# being clipped (which doesn't sanitize nan) and written into the weights.
# Without this, one rare numerically-bad batch leaves the model
# unrecoverable for the rest of training. grad_clip max_norm restored to
# the paper's 35 after Step 1 diagnostic showed pre-clip grad norm
# p99=5.9, max=35.2 over ~3k optimizer steps — the previous tight 1.0
# clip was normalising every step (p50=3.0 >> 1.0), not bounding outliers.
# target_ratio raised from (2, 1e-4) to (5, 1e-4) after Step 2 confirmed
# grad_clip wasn't the plateau constraint — matched_ious re-plateaued at
# 0.46-0.48 from epoch 5, mirroring the pre-Step-1 plateau, so the
# remaining binding constraint is peak LR.
# Step 4: samples_per_gpu raised 1→3 (matches paper's per-GPU batch) and
# cumulative_iters reduced 16→1 (eliminates 16× NaN amplification identified
# in Step 3 diagnostic). Effective batch becomes 12 (4 GPUs × 3 × 1) vs paper's
# 24 (8 GPUs × 3 × 1) — the only remaining unavoidable hardware-driven
# deviation. With idle GPUs available, we match paper's per-GPU statistics
# exactly. lr_config.target_ratio remains (5, 1e-4) for the first run;
# revisit only if epoch-1 matched_ious is much lower than Step 3's 0.344.
optimizer_config = dict(
    type='NanSkipGradientCumulativeOptimizerHook',
    cumulative_iters=1,
    grad_clip=dict(max_norm=35, norm_type=2),
)

# Override cyclic_20e.py's target_ratio=(10, 1e-4). Set to (5, 1e-4) for
# peak LR 5e-4 — half the paper's 10x, conservative headroom over the
# previous Step-2 (2, 1e-4) plateau without going all the way to paper.
# Earlier nan campaigns saw unrecoverable nan at 10x and 5x base, but
# those runs lacked the NanSkipGradientCumulativeOptimizerHook above,
# which now catches per-batch nan/inf before they corrupt weights.
lr_config = dict(
    policy='cyclic',
    target_ratio=(5, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)

two_stage = True
runner = dict(type='TwoStageRunner', max_epochs=10)
num_proposals_test = 300
load_from = './pretrained/r50_fpn_nuImage_pretrained.pth'