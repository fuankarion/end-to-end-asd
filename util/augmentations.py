import random

from PIL import Image
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import hflip


def video_temporal_crop(video_data, crop_ratio):
    # random flip
    if bool(random.getrandbits(1)):
        video_data = [s.transpose(Image.FLIP_LEFT_RIGHT) for s in video_data]

    # random crop
    mid = int(len(video_data) / 2)
    width, height = video_data[mid].size
    f = random.uniform(crop_ratio, 1)
    i, j, h, w = RandomCrop.get_params(video_data[mid], output_size=(int(height*f), int(width*f)))

    video_data = [s.crop(box=(j, i, j+w, i+h)) for s in video_data]
    return video_data


def video_flip(video_data, crop_ratio):
    # random flip
    if bool(random.getrandbits(1)):
        video_data = [hflip(vd) for vd in video_data]

    return video_data
