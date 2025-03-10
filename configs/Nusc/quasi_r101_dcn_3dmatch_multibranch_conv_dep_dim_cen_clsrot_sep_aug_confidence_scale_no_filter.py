# model settings
model = dict(
    type='QuasiDense3DSepUncertainty',
    pretrained='modelzoo://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4, 8],
        anchor_ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='ConvFCBBoxHead',
        num_shared_convs=4,
        num_shared_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=12,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        norm_cfg=None,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=5.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=5.0)
    ),
    bbox_3d_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_3d_head=dict(
        type='ConvFCBBox3DRotSepConfidenceHead',
        num_shared_convs=2,
        num_shared_fcs=0,
        num_dep_convs=4,
        num_dim_convs=4,
        num_rot_convs=4,
        num_2dc_convs=4,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=12,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        center_scale=10,
        reg_class_agnostic=False,
        norm_cfg=None,
        with_depth=True,
        with_uncertainty=True,
        use_uncertainty=True,
        with_dim=True,
        with_rot=True,
        with_2dc=True,
        loss_depth=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0),
        loss_uncertainty=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0),
        loss_dim=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0),
        loss_rot=dict(
            type='RotBinLoss',
            loss_weight=1.0),
        loss_2dc=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
    ),
    embed_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    embed_head=dict(
        type='MultiPos3DTrackHead',
        num_convs=4,
        num_fcs=1,
        embed_channels=256,
        norm_cfg=dict(type='GN', num_groups=32),
        loss_depth=None,
        loss_asso=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            tau=-1),
        loss_iou=dict(
            type='L2Loss',
            sample_ratio=3,
            margin=0.3,
            loss_weight=1.0,
            hard_mining=True))
)  # yapf:disable
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=5,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='CombinedSampler',
            num=512,
            pos_fraction=0.25,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3)),
        pos_weight=-1,
        debug=False),
    embed=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='CombinedSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=3,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3)),
        with_key_pos=True,
        with_ref_pos=True,
        with_ref_neg=True,
        with_key_neg=True)
)  # yapf:disable
test_cfg = dict(
    analyze=False,
    show=False,
    save_txt=dict(bicycle=0, motorcycle=1, pedestrian=2, bus=3, car=4,
                  trailer=5, truck=6, construction_vehicle=7, traffic_cone=8, barrier=9),
    use_3d_center=True,
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100),
    track=dict(
        type='Embedding3DBEVMotionUncertaintyTracker',
        init_score_thr=0.8,
        init_track_id=0,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        motion_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        loc_dim=7,
        with_deep_feat=True,
        with_cats=True,
        with_bbox_iou=True,
        with_depth_ordering=True,
        lstm_name='VeloLSTM',
        lstm_ckpt_name=
        './checkpoints/batch128_min10_seq10_dim7_VeloLSTM_nuscenes_100_linear.pth',
        track_bbox_iou='box3d',
        depth_match_metric='motion',
        tracker_model_name='KalmanBox3DTracker',
        match_metric='cycle_softmax',
        match_algo='greedy')
)  # yapf:disable
# dataset settings
dataset_type = 'BDDVid3DDataset'
data_root = 'data/nuscenes/'
img_norm_cfg = dict(
    mean=[106.07, 127.705, 128.08], std=[73.458, 70.129, 70.463], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=dict(VID=data_root + 'anns/tracking_train.json'),
        img_prefix=dict(VID=data_root + 'samples/'),
        img_scale=(1600, 900),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        with_track=True,
        key_sample_interval=1,
        ref_sample_interval=3,
        ref_sample_sigma=-1,
        track_det_ratio=-1,
        ref_share_flip=True,
        with_3d=True,
        augs_num=3,
        augs_ratio=0.5,
        extra_aug=dict(
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18), )),
    val=dict(
        type=dataset_type,
        ann_file=dict(VID=data_root + 'anns/tracking_val_mini.json'),
        # ann_file=dict(VID=data_root + 'anns/tracking_val.json'),
        img_prefix=dict(VID=data_root + 'samples/'),
        img_scale=(1600, 900),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        with_3d=True,
        with_track=True),
    test=dict(
        type=dataset_type,
        ann_file=dict(VID=data_root + 'anns/tracking_val_mini.json'),
        img_prefix=dict(VID=data_root + 'samples/'),
        img_scale=(1600, 900),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        test_mode=True,
        with_mask=False,
        with_label=True,
        with_3d=True,
        with_track=True))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bboxfc_lr_mult=10.0))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    step=[16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='BDD_VID'))
    ])
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(type='video', interval=24)
