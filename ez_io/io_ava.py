import os
from PIL import Image
from scipy.io import wavfile
import numpy as np
from util.audio_processing import generate_mel_spectrogram


def _pil_loader(path):
    with Image.open(path) as img:
        return img.convert('RGB')


def _cached_pil_loader(path, cache):
    if path in cache.keys():
        return cache[path]

    with Image.open(path) as img:
        rgb = img.convert('RGB')
        cache[path] = rgb
        return rgb


def _fit_audio_clip(audio_clip, sample_rate, video_clip_lenght):
    target_audio_length = int((1.0/27.0)*sample_rate*video_clip_lenght)
    pad_required = int((target_audio_length-len(audio_clip))/2)
    if pad_required > 0:
        audio_clip = np.pad(audio_clip, pad_width=(pad_required, pad_required),
                            mode='reflect')
    if pad_required < 0:
        audio_clip = audio_clip[-1*pad_required:pad_required]

    # TODO There is a +-1 offset here and I dont feel like cheking it
    return audio_clip[0:target_audio_length-1]


def load_v_clip_from_metadata(clip_meta_data, frames_source):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]
    entity_id = clip_meta_data[0][0]

    # Video Frames
    selected_frames = [os.path.join(frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]
    video_data = [_pil_loader(sf) for sf in selected_frames]
    return video_data


def load_v_clip_from_metadata_cache(clip_meta_data, frames_source, cache, silent_fail=False):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]
    entity_id = clip_meta_data[0][0]

    # Video Frames
    selected_frames = [os.path.join(
        frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]
    if silent_fail:
        video_data = [_cached_pil_loader_silent_fail(sf, cache) for sf in selected_frames]
    else:
        video_data = [_cached_pil_loader(sf, cache) for sf in selected_frames]
    return video_data


def load_a_clip_from_metadata(clip_meta_data, frames_source, audio_source,
                              audio_offset, fail_silent=False):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    # Audio File
    audio_file = os.path.join(audio_source, entity_id+'.wav')
    sample_rate, audio_data = wavfile.read(audio_file)

    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(ts_sequence))
    audio_features = generate_mel_spectrogram(audio_clip, sample_rate)

    return audio_features


def load_a_clip_from_metadata_sinc(clip_meta_data, frames_source, audio_source,
                                   audio_offset):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    # keep this var
    selected_frames = [os.path.join(frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]

    # Audio File
    audio_file = os.path.join(audio_source, entity_id+'.wav')
    sample_rate, audio_data = wavfile.read(audio_file)

    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(selected_frames))
    return audio_clip