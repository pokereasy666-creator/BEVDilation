_base_ = ['./bevdilation.py']
# Diagnostic 4 Oracle A config.
#
# Identical to bevdilation.py except the test pipeline also collects GT boxes, so
# the oracle path in the model (SVDB.obtain_bev_mask_gt) can build the perfect
# foreground mask from the SAME pipeline-transformed boxes the model trained on.
# The model's `oracle_fg` flag is the behavioral switch (set by diag4_oracle_eval.py);
# this config only changes data collection. The base config is NOT modified.
#
# The two edits vs the base test pipeline (both inside MultiScaleFlipAug3D.transforms):
#   - DefaultFormatBundle3D(with_label=True)              (was with_label=False)
#   - Collect3D keys += 'gt_bboxes_3d', 'gt_labels_3d'
# Everything else (val pickle, image/point loading, augs) is byte-identical. The six
# variables below are copied verbatim from bevdilation.py so the pipeline can be
# rebuilt here regardless of mmcv base-variable interpolation support; keep in sync.

point_cloud_range = [-54.0, -54.0, -3.0, 54.0, 54.0, 5.0]
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
    'vflip': True,
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

file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5 * 2, 22.5 * 2),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    tran_lim=[0.5, 0.5, 0.5]
)

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
                with_label=True),  # ORACLE EDIT: was with_label=False
            # ORACLE EDIT: also collect GT boxes/labels so the oracle mask can be built
            dict(type='Collect3D',
                 keys=['points', 'img_inputs', 'gt_depth',
                       'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
