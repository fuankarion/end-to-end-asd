import argparse


def get_default_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default='5e-4')
    parser.add_argument('--frmc', default='13')
    parser.add_argument('--ctx', default='2')
    parser.add_argument('--nclp', default='7')
    parser.add_argument('--strd', default='3')
    parser.add_argument('--size', default='160')

    return parser


def unpack_command_line_args(args):
    lr_arg = float(args.lr)
    frames_per_clip = float(args.frmc)
    ctx_size = int(args.ctx)
    n_clips = int(args.nclp)
    strd = int(args.strd)
    img_size = int(args.size)

    return lr_arg, frames_per_clip, ctx_size, n_clips, strd, img_size
