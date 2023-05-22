import os
import csv

def postprocess_speech_label(speech_label):
    speech_label = int(speech_label)
    if speech_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
        speech_label = 0
    return speech_label


def postprocess_entity_label(entity_label):
    entity_label = int(entity_label)
    if entity_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
        entity_label = 0
    return entity_label


def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list


def load_val_video_set():
    files = os.listdir(
        '/ibex/ai/home/alcazajl/ava_active_speakers/csv/gt/ava_activespeaker_test_v1.0')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos


def load_train_video_set():
    files = os.listdir(
        '/ibex/ai/home/alcazajl/ava_active_speakers/csv/gt/ava_activespeaker_train_v1.0')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos


def generate_av_mask(ctx_size, total_len):
    stride = ctx_size + 1
    audio_mask = []
    video_mask = []
    for i in range(0, total_len):
        if i % stride == 0:
            audio_mask.append(i)
        else:
            video_mask.append(i)
    return audio_mask, video_mask


def generate_avs_mask(ctx_size, total_len):
    stride = ctx_size + 2
    audio_mask = []
    sync_mask = []
    video_mask = []
    for i in range(0, total_len):
        if i % stride == 0:
            audio_mask.append(i)
        elif i % stride == 1:
            sync_mask.append(i)
        else:
            video_mask.append(i)
    return audio_mask, sync_mask, video_mask