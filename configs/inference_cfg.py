model_type='ae'
inference_param=dict(
    exp_name = 'gsinpaint_exp1',   # TODO
    model_path='',   # TODO
    sample_dir = 'tests', # TODO
    mask = True,
)

data_param = dict(
    path = '',       # TODO
    mask_type = [2,4],

    min_ratio=0.2, 
    max_ratio=0.4,

    is_train=False,
    shuffle=True,
    num_workers=8,
    bs=16,
)


model_param = dict(
        is_train=False,
        image_size=(256,256),
        patch_size=(16,16),
        hidden_dim=12,
        use_dino_cls = True,
        use_dino_pred_loss = True,
        gaussian_per_patch=324,
        encoder_type = 'resnet',
        encoder_args=dict(
            use_skip=True,
            norm_layer = 'gn',
            up_norm_layer = 'gn',
        ),
        overlap=True,
        overlap_pad = 1,
        condition_type = 'direct',
    )
