_base_ = '../../../../mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)


model = dict(
    roi_head=dict(
        type='CascadeRoIHead',
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=1,),
            dict(type='Shared2FCBBoxHead', num_classes=1,),
            dict(type='Shared2FCBBoxHead', num_classes=1,)
        ],
        mask_head=dict(type='FCNMaskHead', num_classes=1),))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.2),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=10,
    #             g_shift_limit=10,
    #             b_shift_limit=10,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=3, p=1.0),
    #         dict(type='MedianBlur', blur_limit=3, p=1.0)
    #     ],
    #     p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('wrinkle',)
data = dict(
    train=dict(
        pipeline=train_pipeline,
        img_prefix='../dataset/G1/train/',
        classes=classes,
        ann_file='../dataset/G1/train.json'),
    val=dict(
        img_prefix='../dataset/G1/val/',
        classes=classes,
        ann_file='../dataset/G1/val.json'),
    test=dict(
        img_prefix='../dataset/R_test/',
        classes=classes,
        ann_file='../dataset/R_test/r_test.json'))

checkpoint_config = dict(interval=1)
log_config = dict(interval=50,)

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)

load_from='./checkpoints/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth'