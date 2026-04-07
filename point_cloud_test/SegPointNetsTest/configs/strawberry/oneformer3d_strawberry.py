_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

num_channels = 16
num_instance_classes = 3
num_semantic_classes = 4
class_names = ['background', 'ripe', 'half-ripe', 'unripe', 'unlabeled']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names,
    dataset_name='StrawberryBush')

model = dict(
    type='ScanNetOneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.01,
    num_classes=num_instance_classes,
    min_spatial_shape=64,
    query_thr=0.5,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels, num_channels * 2],
        return_blocks=True),
    decoder=dict(
        type='ScanNetQueryDecoder',
        num_layers=3,
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=16,
        d_model=128,
        num_heads=8,
        hidden_dim=512,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False),
    criterion=dict(
        type='ScanNetUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',
            ignore_index=num_semantic_classes,
            loss_weight=0.5),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='SparseMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)],
                topk=1),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.1,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=64,
        inst_score_thr=0.0,
        pan_score_thr=0.4,
        npoint_thr=1,
        obj_normalization=True,
        sp_score_thr=0.2,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0],
        num_sem_cls=num_semantic_classes))

dataset_type = 'StrawberrySegDataset'
data_root = '/workspace/data_exports/strawberry_benchmark/oneformer3d'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.5, 0.5],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.02, 0.02, 0.02],
        shift_height=False),
    dict(type='NormalizePointsColor_', color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='AddSuperPointAnnotations',
        num_classes=num_semantic_classes,
        stuff_classes=[0],
        merge_non_stuff_cls=False),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='PointSegClassMapping'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor_', color_mean=[127.5, 127.5, 127.5]),
            dict(
                type='AddSuperPointAnnotations',
                num_classes=num_semantic_classes,
                stuff_classes=[0],
                merge_non_stuff_cls=False),
        ]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_oneformer3d_infos_train.pkl',
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=train_pipeline,
        ignore_index=num_semantic_classes,
        scene_idxs=None,
        test_mode=False))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_oneformer3d_infos_val.pkl',
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_oneformer3d_infos_test.pkl',
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=[1, 2, 3],
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=[0, 1, 2, 3],
    inst_mapping=[1, 2, 3],
    metric_meta=metric_meta)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=48, power=0.9)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(checkpoint=dict(interval=4, max_keep_ckpts=4))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=48,
    val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
