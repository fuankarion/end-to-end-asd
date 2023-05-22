import os
import torch

import experiment_config as exp_conf
import util.custom_transforms as ct

from torchvision import transforms
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from ez_io.logging import setup_optim_outputs
from datasets.graph_datasets import IndependentGraphDatasetETE3D
from models.graph_layouts import get_spatial_connection_pattern
from models.graph_layouts import get_temporal_connection_pattern
from optimization.optimization_amp import optimize_easee
from util.command_line import unpack_command_line_args, get_default_arg_parser


if __name__ == '__main__':
    # Parse Command line args
    command_line_args = get_default_arg_parser().parse_args()
    lr_arg, frames_per_clip, ctx_size, n_clips, strd, img_size = unpack_command_line_args(command_line_args)

    # Connection pattern
    scp = get_spatial_connection_pattern(ctx_size, n_clips)
    tcp = get_temporal_connection_pattern(ctx_size, n_clips)

    opt_config = exp_conf.EASEE_R3D_18_4lvl_params
    easee_config = exp_conf.EASEE_R3D_18_inputs

    # Data Transforms
    image_size = (img_size, img_size)
    video_train_transform = transforms.Compose([transforms.Resize(image_size), ct.video_train])
    video_val_transform = transforms.Compose([transforms.Resize(image_size), ct.video_val])

    # output config
    model_name = 'easee_R3D_50' + \
                 '_clip' + str(frames_per_clip) + \
                 '_ctx' + str(ctx_size) + \
                 '_len' + str(n_clips) + \
                 '_str' + str(strd)
    log, target_models = setup_optim_outputs(easee_config['models_out'],
                                             easee_config, model_name)

   # Create Network and offload to GPU
    pretrain_weightds_path = easee_config['video_pretrain_weights']
    ez_net = easee_config['backbone'](pretrain_weightds_path)

    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if has_cuda else 'cpu')
    print('Cuda info ',has_cuda, device)
    ez_net.to(device)

    # Optimization config
    criterion = easee_config['criterion']
    optimizer = easee_config['optimizer'](ez_net.parameters(), lr=easee_config['learning_rate'])
    scheduler = MultiStepLR(optimizer, milestones=[6, 8], gamma=0.1)

    # Data Paths
    video_train_path = os.path.join(easee_config['video_dir'], 'train')
    audio_train_path = os.path.join(easee_config['audio_dir'], 'train')
    video_val_path = os.path.join(easee_config['video_dir'], 'val')
    audio_val_path = os.path.join(easee_config['audio_dir'], 'val')

    # Dataloaders
    d_train = IndependentGraphDatasetETE3D(audio_train_path, 
                                           video_train_path,
                                           easee_config['csv_train_full'], 
                                           n_clips,
                                           strd, 
                                           ctx_size,
                                           frames_per_clip,
                                           scp, tcp, 
                                           video_train_transform,
                                           do_video_augment=True, 
                                           crop_ratio=0.95)
    d_val = IndependentGraphDatasetETE3D(audio_val_path, 
                                         video_val_path,
                                         easee_config['csv_val_full'], 
                                         n_clips,
                                         strd, 
                                         ctx_size, 
                                         frames_per_clip,
                                         scp, tcp, 
                                         video_val_transform,
                                         do_video_augment=False)

    dl_train = DataLoader(d_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'],
                          pin_memory=True)
    dl_val = DataLoader(d_val, batch_size=opt_config['batch_size'],
                        shuffle=True, num_workers=opt_config['threads'],
                        pin_memory=True)

    # Optimization loop
    model = optimize_easee(ez_net, dl_train, dl_val, device,
                           criterion, optimizer, scheduler,
                           num_epochs=opt_config['epochs'],
                           spatial_ctx_size=ctx_size,
                           time_len=n_clips,
                           models_out=target_models, log=log)
