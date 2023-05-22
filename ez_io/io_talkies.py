import os

def load_av_clip_from_metadata_talkies(speaker_data, mid_index, half_clip_lenght,
                                       video_root, audio_root):
    #midone = speaker_data[mid_index]

    idx_sequence = [i for i in range(
        mid_index-half_clip_lenght, mid_index+half_clip_lenght+1)]
    idx_sequence = [idx if idx >= 0 else 0 for idx in idx_sequence]
    idx_sequence = [idx if idx < len(speaker_data) else len(
        speaker_data)-1 for idx in idx_sequence]

    frame_sequence = [speaker_data[idx] for idx in idx_sequence]
    frame_sequence = [os.path.join(video_root, str(f)+'.jpg')
                      for f in frame_sequence]
    #print(frame_sequence)
    min_ts = float(os.path.basename(frame_sequence[0])[:-4])
    max_ts = float(os.path.basename(frame_sequence[-1])[:-4])

    # Video Frames
    video_data = [_pil_loader(sf) for sf in frame_sequence]

    # Audio File
    audio_file = os.path.join(audio_root+'.wav')
    sample_rate, audio_data = wavfile.read(audio_file)

    audio_start = int(min_ts*sample_rate)
    audio_end = int(max_ts*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    if len(audio_clip) == 0:
        #print('S0', speaker_data, idx_sequence, frame_sequence)
        audio_clip = np.zeros((int(0.3*sample_rate)))

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(frame_sequence))
    audio_features = generate_mel_spectrogram(audio_clip, sample_rate)

    return video_data, audio_features


def load_a_clip_from_metadata_talkies(video_id, clip_meta_data, audio_source,
                                      audio_offset, fail_silent=False):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    # Audio File
    audio_file = os.path.join(audio_source, video_id+'.wav')
    try:
        sample_rate, audio_data = wavfile.read(audio_file)
    except:
        sample_rate = 16000
        audio_data = np.zeros((int(2*sample_rate)))


    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    # TODO FIX
    if len(audio_clip) == 0:
        print('S0', video_id, min_ts, max_ts, len(audio_data))
        audio_clip = np.zeros((int(0.3*sample_rate)))

    #print('bbb', min_ts, max_ts, audio_start,
    #      audio_end, len(audio_data), len(audio_clip))
    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(ts_sequence))
    audio_features = generate_mel_spectrogram(audio_clip, sample_rate)

    return audio_features


def load_v_clip_from_metadata_cache_talkies(video_id, clip_meta_data, frames_source, cache, silent_fail=False):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]
    entity_id = clip_meta_data[0][0]
    entity_id = entity_id.replace(' ', '_')

    # Video Frames
    selected_frames = [os.path.join(
        frames_source, video_id, entity_id, ts+'.jpg') for ts in ts_sequence]
    if silent_fail:
        video_data = [_cached_pil_loader_silent_fail(
            sf, cache) for sf in selected_frames]
    else:
        video_data = [_cached_pil_loader(sf, cache) for sf in selected_frames]
    return video_data


def load_talkies_clip_from_metadata(clip_meta_data, frames_source,
                                    audio_file):

    selected_frames = [os.path.join(
        frames_source, str(ts)+'.jpg') for ts in clip_meta_data]
    video_data = [_pil_loader(sf) for sf in selected_frames]

    # audio data
    sample_rate, audio_data = wavfile.read(audio_file)
    audio_start = int(clip_meta_data[0]*sample_rate)
    audio_end = int(clip_meta_data[-1]*sample_rate)
    audio_clip = audio_data[audio_start:audio_end+1]

    l_pad_size = -1  # yup -1 dont switch it
    r_pad_size = -1
    for cmd in clip_meta_data:
        if cmd == clip_meta_data[0]:
            l_pad_size = l_pad_size + 1
        if cmd == clip_meta_data[-1]:
            r_pad_size = r_pad_size + 1

    l_pad_size = int(l_pad_size*(1/30)*sample_rate)
    r_pad_size = int(r_pad_size*(1/30)*sample_rate)

    audio_clip = np.pad(audio_clip, (l_pad_size, r_pad_size), mode='reflect')
    audio_features = generate_mel_spectrogram(audio_clip, sample_rate)
    return video_data, audio_features