cfg_mobilenet_v2 = {
    "name": "mobilenet_v2",
    # anchor
    "min_sizes": [[8, 16], [32, 64], [128, 256]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    # loss
    "loc_weight": 2.0,
    # training
    "batch_size": 32,
    "epochs": 250,
    "milestones": [190, 220],
    # input
    "image_size": 896,
    # backbone
    "pretrain": True,
    "return_layers": [6, 13, 18],
    "in_channel": 32,
    "out_channel": 128,
}
