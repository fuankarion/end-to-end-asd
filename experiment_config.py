import torch.nn as nn
import torch.optim as optim

import models.graph_models as g3d


EASEE_R3D_18_inputs = {
    # input files
    'csv_train_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_val_augmented.csv',
    'csv_test_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_test_augmented.csv',

    # Data config
    'audio_dir': '/Dataset/ava_active_speaker/instance_wavs_time/',
    'video_dir': '/Dataset/ava_active_speaker/instance_crops_time/',
    'models_out': '/home/alcazajl/Models/ASC2/tan3d/',  # save directory

    # Pretrained Weights
    'video_pretrain_weights': '/home/alcazajl/Models/Pretrained/R3D/r3d18_K_200ep.pth'
}

EASEE_R3D_50_inputs = {
    # input files
    'csv_train_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_train_augmented.csv',
    'csv_val_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_val_augmented.csv',
    'csv_test_full': '/Dataset/ava_active_speaker/csv/ava_activespeaker_test_augmented.csv',

    # Data config
    'audio_dir': '/Dataset/ava_active_speaker/instance_wavs_time/',
    'video_dir': '/Dataset/ava_active_speaker/instance_crops_time/',
    'models_out': '/home/alcazajl/Models/ASC2/tan3d/',  # save directory

    # Pretrained Weights
    'video_pretrain_weights': '/home/alcazajl/Models/Pretrained/R3D/r3d50_K_200ep.pth'
}


EASEE_R3D_18_4lvl_params = {
    # Net Arch
    'backbone': g3d.R3D18_4lvlGCN,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-4,
    'epochs': 15,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 17,
    'threads': 8
}


EASEE_R3D_50_4lvl_params = {
    # Net Arch
    'backbone': g3d.R3D50_4lvlGCN,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-4,
    'epochs': 15,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 17,
    'threads': 8
}
