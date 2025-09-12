model_type = 'ae'
train_param=dict(
    exp_name = 'gsinpaint_exp1',  # TODO

    lr=2e-4,
    gan_lr = 2e-4,  
    d_zero_step = 1, # by epoch
    d_warmup_step = 1,  # by epoch

    epoch=400,  
    milestones = [350,380],

    resume=False, 
    model_path=None,

    mask = True,

    # visualize related
    wandb = True,
    wandb_proj = 'gsinpaint_exp1',   # TODO
    wandb_name = '0',
    log_step=10,
    save_step=5000,
    sample_step = 200,
    sample_dir = 'samples',  # TODO
    output_dir = 'outputs',  # TODO

    use_multi_mask_ratio = True,
    ratio_interp = 20, 
    min_ratio = 0.2,
    max_ratio = 0.6,

    loss_type = dict(
        reconstruction='full_l1',
        gan='hinge',
        gan_iter_step = 1,
        gan_gradient_penalty = False,
        r1_weight=10, 
        perceptual_loss='lpips',

        partial_scale = 0,
        recons_scale = 1,
        perceptual_scale=3,
        gan_scale = 0.3,
        dino_pred_scale = 1,
    ),

)


data_param = dict(
    path = '',   # TODO
    mask_type = [2,4],

    is_train=True,
    shuffle=True,
    num_workers=4, 
    bs=8,
)

model_param = dict(
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
