custom_imports = dict(imports=['geospatial_fm'])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
dataset_type = 'GeospatialDataset'
data_root = '/home/ubuntu/files/'
num_frames = 1
img_size = 224
num_workers = 4
samples_per_gpu = 4
img_norm_cfg = dict(
    means=[
        0.033349706741586264, 0.05701185520536176, 0.05889748132001316,
        0.2323245113436119, 0.1972854853760658, 0.11944914225186566
    ],
    stds=[
        0.02269135568823774, 0.026807560223070237, 0.04004109844362779,
        0.07791732423672691, 0.08708738838140137, 0.07241979477437814
    ])
bands = [0, 1, 2, 3, 4, 5]
tile_size = 224
orig_nsize = 512
crop_size = (224, 224)
img_suffix = '_merged.tif'
seg_map_suffix = '.mask.tif'
ignore_index = -1
image_nodata = -9999
image_nodata_replace = 0
image_to_float32 = True
pretrained_weights_path = '/home/ubuntu/hls-foundation-os/Prithvi_100M.pt'
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1
output_embed_dim = 768
max_intervals = 10000
evaluation_interval = 1000
experiment = 'scar'
project_dir = '/home/ubuntu/project/'
work_dir = '/home/ubuntu/project/scar'
save_path = '/home/ubuntu/project/scar'
train_pipeline = [
    dict(
        type='LoadGeospatialImageFromFile',
        to_float32=True,
        channels_last=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=False),
    dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
    dict(
        type='TorchNormalize',
        means=[
            0.033349706741586264, 0.05701185520536176, 0.05889748132001316,
            0.2323245113436119, 0.1972854853760658, 0.11944914225186566
        ],
        stds=[
            0.02269135568823774, 0.026807560223070237, 0.04004109844362779,
            0.07791732423672691, 0.08708738838140137, 0.07241979477437814
        ]),
    dict(type='TorchRandomCrop', crop_size=(224, 224)),
    dict(type='Reshape', keys=['img'], new_shape=(6, 1, 224, 224)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, 224, 224)),
    dict(
        type='CastTensor',
        keys=['gt_semantic_seg'],
        new_type='torch.LongTensor'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type='LoadGeospatialImageFromFile',
        to_float32=True,
        channels_last=True),
    dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
    dict(type='ToTensor', keys=['img']),
    dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
    dict(
        type='TorchNormalize',
        means=[
            0.033349706741586264, 0.05701185520536176, 0.05889748132001316,
            0.2323245113436119, 0.1972854853760658, 0.11944914225186566
        ],
        stds=[
            0.02269135568823774, 0.026807560223070237, 0.04004109844362779,
            0.07791732423672691, 0.08708738838140137, 0.07241979477437814
        ]),
    dict(
        type='Reshape',
        keys=['img'],
        new_shape=(6, 1, -1, -1),
        look_up=dict({
            '2': 1,
            '3': 2
        })),
    dict(type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
    dict(
        type='CollectTestList',
        keys=['img'],
        meta_keys=[
            'img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename',
            'ori_filename', 'img', 'img_shape', 'ori_shape', 'pad_shape',
            'scale_factor', 'img_norm_cfg'
        ])
]
CLASSES = ('Unburnt land', 'Burn scar')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='GeospatialDataset',
        CLASSES=('Unburnt land', 'Burn scar'),
        data_root='/home/ubuntu/files/',
        img_dir='training',
        ann_dir='training',
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        pipeline=[
            dict(
                type='LoadGeospatialImageFromFile',
                to_float32=True,
                channels_last=True),
            dict(type='LoadGeospatialAnnotations', reduce_zero_label=False),
            dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
            dict(type='RandomFlip', prob=0.5),
            dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    0.033349706741586264, 0.05701185520536176,
                    0.05889748132001316, 0.2323245113436119,
                    0.1972854853760658, 0.11944914225186566
                ],
                stds=[
                    0.02269135568823774, 0.026807560223070237,
                    0.04004109844362779, 0.07791732423672691,
                    0.08708738838140137, 0.07241979477437814
                ]),
            dict(type='TorchRandomCrop', crop_size=(224, 224)),
            dict(type='Reshape', keys=['img'], new_shape=(6, 1, 224, 224)),
            dict(
                type='Reshape',
                keys=['gt_semantic_seg'],
                new_shape=(1, 224, 224)),
            dict(
                type='CastTensor',
                keys=['gt_semantic_seg'],
                new_type='torch.LongTensor'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        ignore_index=-1),
    val=dict(
        type='GeospatialDataset',
        CLASSES=('Unburnt land', 'Burn scar'),
        data_root='/home/ubuntu/files/',
        img_dir='validation',
        ann_dir='validation',
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        pipeline=[
            dict(
                type='LoadGeospatialImageFromFile',
                to_float32=True,
                channels_last=True),
            dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    0.033349706741586264, 0.05701185520536176,
                    0.05889748132001316, 0.2323245113436119,
                    0.1972854853760658, 0.11944914225186566
                ],
                stds=[
                    0.02269135568823774, 0.026807560223070237,
                    0.04004109844362779, 0.07791732423672691,
                    0.08708738838140137, 0.07241979477437814
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(6, 1, -1, -1),
                look_up=dict({
                    '2': 1,
                    '3': 2
                })),
            dict(
                type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
            dict(
                type='CollectTestList',
                keys=['img'],
                meta_keys=[
                    'img_info', 'seg_fields', 'img_prefix', 'seg_prefix',
                    'filename', 'ori_filename', 'img', 'img_shape',
                    'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
                ])
        ],
        ignore_index=-1),
    test=dict(
        type='GeospatialDataset',
        CLASSES=('Unburnt land', 'Burn scar'),
        data_root='/home/ubuntu/files/',
        img_dir='validation',
        ann_dir='validation',
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        pipeline=[
            dict(
                type='LoadGeospatialImageFromFile',
                to_float32=True,
                channels_last=True),
            dict(type='BandsExtract', bands=[0, 1, 2, 3, 4, 5]),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    0.033349706741586264, 0.05701185520536176,
                    0.05889748132001316, 0.2323245113436119,
                    0.1972854853760658, 0.11944914225186566
                ],
                stds=[
                    0.02269135568823774, 0.026807560223070237,
                    0.04004109844362779, 0.07791732423672691,
                    0.08708738838140137, 0.07241979477437814
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(6, 1, -1, -1),
                look_up=dict({
                    '2': 1,
                    '3': 2
                })),
            dict(
                type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
            dict(
                type='CollectTestList',
                keys=['img'],
                meta_keys=[
                    'img_info', 'seg_fields', 'img_prefix', 'seg_prefix',
                    'filename', 'ori_filename', 'img', 'img_shape',
                    'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
                ])
        ],
        ignore_index=-1))
optimizer = dict(type='Adam', lr=1.3e-05, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
checkpoint_config = dict(
    by_epoch=True, interval=10, out_dir='/home/ubuntu/project/scar')
evaluation = dict(
    interval=1000,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
    by_epoch=False)
loss_func = dict(
    type='DiceLoss', use_sigmoid=False, loss_weight=1, ignore_index=-1)
runner = dict(type='IterBasedRunner', max_iters=10000)
workflow = [('train', 1)]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='TemporalEncoderDecoder',
    frozen_backbone=False,
    backbone=dict(
        type='TemporalViTEncoder',
        pretrained='/home/ubuntu/hls-foundation-os/Prithvi_100M.pt',
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_pix_loss=False),
    neck=dict(
        type='ConvTransformerTokensToEmbeddingNeck',
        embed_dim=768,
        output_embed_dim=768,
        drop_cls_token=True,
        Hp=14,
        Wp=14),
    decode_head=dict(
        num_classes=2,
        in_channels=768,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1,
            ignore_index=-1)),
    auxiliary_head=dict(
        num_classes=2,
        in_channels=768,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1,
            ignore_index=-1)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(112, 112), crop_size=(224, 224)))
gpu_ids = range(0, 1)
auto_resume = False
