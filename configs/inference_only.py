# _base_ = [
#     '../../../../_base_/default_runtime.py',
#     # '../../../../_base_/datasets/coco.py'
#     '../../../../_base_/datasets/xray.py'
# ]
_base_ = [
    '_base_/default_runtime.py',
    '_base_/datasets/xray.py'
]
checkpoint_config = dict(interval=25)
evaluation = dict(interval=50, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=0.0015,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200, 260])
total_epochs = 300
# channel_cfg = dict(
#     dataset_joints=17, # modified to 4
#     dataset_channel=[ # modified to [0,1,2,3]
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
#     ],
#     inference_channel=[ # modified to [0,1,2,3]
#         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
#     ])
channel_cfg = dict(
    dataset_joints=24,  # modified to 4
    dataset_channel=[  # modified to [0,1,2,3]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    ],
    inference_channel=[  # modified to [0,1,2,3]
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ])

data_cfg = dict(
    image_size=512,  # modify
    base_size=256,  # modify
    base_sigma=2,  # maybe modify，是高斯核半径
    heatmap_size=[128, 256],  # maybe modify
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=2,  # number of heatmap_size
    scale_aware_sigma=False,  # maybe True, try
)

# model settings
model = dict(
    type='AssociativeEmbedding',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w32-36af842e.pth',  # modified to higherhrnet
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='AEHigherResolutionHead',
        in_channels=32,
        num_joints=24,  # modified to 24
        tag_per_joint=True,
        extra=dict(final_conv_kernel=1, ),
        num_deconv_layers=1,
        num_deconv_filters=[32],
        num_deconv_kernels=[4],
        num_basic_blocks=4,
        cat_output=[True],
        with_ae_loss=[True, False],
        loss_keypoint=dict(
            type='MultiLossFactory',
            num_joints=24,  # modified to 24
            num_stages=2,
            ae_loss_type='exp',
            with_ae_loss=[True, False],
            push_loss_factor=[0.001, 0.001],
            pull_loss_factor=[0.001, 0.001],
            with_heatmaps_loss=[True, True],
            heatmaps_loss_factor=[1.0, 1.0])),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,  # modified to 10? 7? fine tune?
        scale_factor=[1],
        with_heatmaps=[True, True],
        with_ae=[True, False],
        project2image=True,
        align_corners=False,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        flip_test=True))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=30,  # modified to 10? 7? fine tune?
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

# data_root = 'data/coco' # modified
# data_root = '/home/myy/jingzhui/Xray/Xray_data_prepare/Xray_pic'
data_root = '/home/myy/jingzhui_2411/images_all'
data = dict(
    # workers_per_gpu=2,
    workers_per_gpu=0,
    train_dataloader=dict(samples_per_gpu=8),  # batch_size
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpCocoDataset',
        # ann_file=f'{data_root}/annotations/person_keypoints_train2017.json', # modified
        # ann_file='/home/myy/jingzhui/Xray/Xray_data_prepare/annotation1500_xin_6_zhuiti_train_coco.json',
        # ann_file='/home/myy/jingzhui/Xray/Xray_data_prepare/annotation1500_xin_6_zhuiti_train_coco_new.json',
        ann_file = '/home/myy/jingzhui_2411/annotation_coco.json',
        img_prefix=f'{data_root}/',  # modify
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    val=dict(
        type='BottomUpCocoDataset',
        # ann_file=f'{data_root}/annotations/person_keypoints_val2017.json', # modified
        # ann_file='/home/myy/jingzhui/Xray/Xray_data_prepare/annotation1500_xin_6_zhuiti_test_coco.json',
        # ann_file='/home/myy/jingzhui/Xray/Xray_data_prepare/annotation1500_xin_6_zhuiti_test_coco_new.json',
        ann_file = '/home/myy/jingzhui_2411/annotation_coco.json',
        img_prefix=f'{data_root}',  # modified
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}
    ),
    test=dict(
        type='BottomUpCocoDataset',
        # ann_file='/home/myy/jingzhui/Xray/Xray_data_prepare/annotation1500_xin_6_zhuiti_train_zhengce_coco.json',
        # ann_file='/home/myy/jingzhui/Xray/Xray_data_prepare/annotation1500_xin_6_zhuiti_train_houshen_coco.json',
        ann_file='/home/myy/jingzhui_2411/annotation_coco.json',
        # ann_file='/home/myy/jingzhui/Xray/Xray_data_prepare/annotation_waibu_test_qianqu_coco.json',
        # ann_file='/home/myy/jingzhui/Xray/Xray_data_prepare/annotation_waibu_test_houshen_coco.json',
        # img_prefix=f'{data_root}_train_zhengce/',  # modified
        # img_prefix=f'{data_root}_train_houshen/',  # modified
        img_prefix=f'{data_root}',  # modified
        # img_prefix=f'{data_root}_qianqu/',  # modified
        # img_prefix=f'{data_root}_houshen/',  # modified
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}
    ),
)
find_unused_parameters = False
