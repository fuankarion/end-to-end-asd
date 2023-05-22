import os
import math
import torch
import random

import numpy as np
import ez_io.io_ava as io
import util.clip_utils as cu

from torch_geometric.data import Data, Dataset
from ez_io.file_util import csv_to_list, postprocess_speech_label, postprocess_entity_label
from util.augmentations import video_temporal_crop, video_corner_crop


class GraphContextualDataset(Dataset):
    def __init__(self):
        # In memory data
        self.entity_data = {}
        self.speech_data = {}
        self.ts_to_entity = {}

        self.entity_list = []

    def get_speaker_context(self, video_id, target_entity_id, center_ts, ctx_len):
        # Get contex and exclude self
        context_entities = list(self.ts_to_entity[video_id][center_ts])
        random.shuffle(context_entities)
        context_entities.remove(target_entity_id)

        # but remeber you must include self
        if not context_entities:
            context_entities.insert(0, target_entity_id)
            while len(context_entities) < ctx_len:  # self is context
                context_entities.append(random.choice(context_entities))
        elif len(context_entities) < ctx_len:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            while len(context_entities) < ctx_len:
                context_entities.append(random.choice(context_entities[1:]))
        else:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            context_entities = context_entities[:ctx_len]

        return context_entities

    def search_ts_in_meta_data(self, entity_metadata, ts):
        for idx, em in enumerate(entity_metadata):
            if em[1] == ts:
                return idx
        raise Exception('Bad Context')

    def _cache_entity_data(self, csv_file_path):
        entity_set = set()

        csv_data = csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        for csv_row in csv_data:
            video_id = csv_row[0]
            entity_id = csv_row[-3]
            timestamp = csv_row[1]

            speech_label = postprocess_speech_label(csv_row[-2])
            entity_label = postprocess_entity_label(csv_row[-2])
            minimal_entity_data = (entity_id, timestamp, entity_label)

            # Store minimal entity data
            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}
            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
                entity_set.add((video_id, entity_id))
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

            # Store speech meta-data
            if video_id not in self.speech_data.keys():
                self.speech_data[video_id] = {}
            if timestamp not in self.speech_data[video_id].keys():
                self.speech_data[video_id][timestamp] = speech_label

            # Max operation yields if someone is speaking.
            new_speech_label = max(
                self.speech_data[video_id][timestamp], speech_label)
            self.speech_data[video_id][timestamp] = new_speech_label

        return entity_set

    def _entity_list_postprocessing(self, entity_set):
        print('Initial', len(entity_set))

        # filter out missing data on disk
        print('video_root',self.video_root)
        all_disk_data = set(os.listdir(self.video_root))
        for video_id, entity_id in entity_set.copy():
            if entity_id not in all_disk_data:
                entity_set.remove((video_id, entity_id))
        print('Pruned not in disk', len(entity_set))

        for video_id, entity_id in entity_set.copy():
            dir = os.path.join(self.video_root, entity_id)
            if len(os.listdir(dir)) != len(self.entity_data[video_id][entity_id]):
                entity_set.remove((video_id, entity_id))

        print('Pruned not complete', len(entity_set))
        self.entity_list = sorted(list(entity_set))

        # Allocate Simultanous Entities
        for video_id, entity_id in entity_set:
            if video_id not in self.ts_to_entity.keys():
                self.ts_to_entity[video_id] = {}

            ent_min_data = self.entity_data[video_id][entity_id]
            for ed in ent_min_data:
                timestamp = ed[1]
                if timestamp not in self.ts_to_entity[video_id].keys():
                    self.ts_to_entity[video_id][timestamp] = []
                self.ts_to_entity[video_id][timestamp].append(entity_id)


class GraphDatasetETE(GraphContextualDataset):
    def __init__(self, audio_root, video_root, csv_file_path,
                 context_size, clip_lenght, connection_pattern,
                 video_transform=None, do_video_augment=False,
                 crop_ratio=0.8, norm_audio=False):
        super().__init__()

        # Data directories
        self.audio_root = audio_root
        self.video_root = video_root

        # Post-processing
        self.crop_ratio = crop_ratio
        self.video_transform = video_transform
        self.do_video_augment = do_video_augment

        # Graph Layout
        self.context_size = context_size

        # Node config
        self.norm_audio = norm_audio
        self.half_clip_length = math.floor(clip_lenght/2)

        # Cache data
        entity_set = self._cache_entity_data(csv_file_path)
        self._entity_list_postprocessing(entity_set)

        # Edge Config
        src_edges = connection_pattern['src']
        dst_edges = connection_pattern['dst']
        self.batch_edges = torch.tensor([src_edges, dst_edges], dtype=torch.long)

        # Replicate entity list
        self.entity_list.extend(self.entity_list)
        self.avg_time = []

    def __len__(self):
        return int(len(self.entity_list)/1)

    def get_audio_size(self,):
        video_id, entity_id = self.entity_list[0]
        entity_metadata = self.entity_data[video_id][entity_id]
        audio_offset = float(entity_metadata[0][1])
        mid_index = random.randint(0, len(entity_metadata)-1)

        clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index, self.half_clip_length)
        audio_data = io.load_a_clip_from_metadata(clip_meta_data, self.video_root, self.audio_root, audio_offset)
        return np.float32(audio_data).shape

    def _get_scene_video_data(self, video_id, entity_id, mid_index):
        orginal_entity_metadata = self.entity_data[video_id][entity_id]
        time_ent = orginal_entity_metadata[mid_index][1]
        context = self.get_speaker_context(
            video_id, entity_id, time_ent, self.context_size)

        video_data = []
        targets = []
        for ctx_entity in context:
            entity_metadata = self.entity_data[video_id][ctx_entity]
            ts_idx = self.search_ts_in_meta_data(entity_metadata, time_ent)
            target_ctx = int(entity_metadata[ts_idx][-1])

            clip_meta_data = cu.generate_clip_meta(entity_metadata, ts_idx, self.half_clip_length)
            video_data.append(io.load_v_clip_from_metadata(clip_meta_data, self.video_root))
            targets.append(target_ctx)

        if self.do_video_augment:
            video_data = [video_temporal_crop(vd, self.crop_ratio) for vd in video_data]

        if self.video_transform is not None:
            for vd_idx, vd in enumerate(video_data):
                tensor_vd = [self.video_transform(f) for f in vd]
                video_data[vd_idx] = tensor_vd

        video_data = [torch.cat(vd, dim=0) for vd in video_data]
        return video_data, targets

    def _get_audio_data(self, video_id, entity_id, mid_index):
        entity_metadata = self.entity_data[video_id][entity_id]
        audio_offset = float(entity_metadata[0][1])
        midone = entity_metadata[mid_index]
        target_audio = self.speech_data[video_id][midone[1]]

        clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index, self.half_clip_length)
        audio_data = io.load_a_clip_from_metadata(clip_meta_data, self.video_root, self.audio_root, audio_offset)
        return np.float32(audio_data), target_audio

    def __getitem__(self, index):
        video_id, entity_id = self.entity_list[index]
        target_entity_metadata = self.entity_data[video_id][entity_id]
        target_index = random.randint(0, len(target_entity_metadata)-1)

        # get av data
        video_data, target_v = self._get_scene_video_data(video_id, entity_id, target_index)
        audio_data, target_a = self._get_audio_data(video_id, entity_id, target_index)

        if self.norm_audio:
            audio_data = (audio_data+3.777757875102366)/186.4988690376491

        # Fill targets
        target_set = []
        target_set.append(target_a)
        for tv in target_v:
            target_set.append(tv)

        # Feature data
        feature_set = torch.zeros((len(video_data)+1, video_data[0].size(0), video_data[0].size(1), video_data[0].size(2)))
        audio_data = torch.from_numpy(audio_data)
        feature_set[0, 0, :audio_data.size(
            1), :audio_data.size(2)] = audio_data
        for i in range(self.context_size):
            feature_set[i+1, ...] = video_data[i]

        return Data(x=feature_set, edge_index=self.batch_edges, y=torch.tensor(target_set))


class IndependentGraphDatasetETE3D(GraphDatasetETE):
    def __init__(self, audio_root, video_root, csv_file_path,
                 graph_time_steps, stride, context_size, clip_lenght,
                 spatial_connection_pattern, temporal_connection_pattern,
                 video_transform=None, do_video_augment=False, crop_ratio=0.95,
                 norm_audio=False):
        super().__init__(audio_root, video_root, csv_file_path,
                         context_size, clip_lenght,
                         spatial_connection_pattern, video_transform,
                         do_video_augment, crop_ratio, norm_audio)

        # Superclass Edge Config
        self.batch_edges = None

        spatial_src_edges = spatial_connection_pattern['src']
        spatial_dst_edges = spatial_connection_pattern['dst']
        self.spatial_batch_edges = torch.tensor(
            [spatial_src_edges, spatial_dst_edges], dtype=torch.long)

        temporal_src_edges = temporal_connection_pattern['src']
        temporal_dst_edges = temporal_connection_pattern['dst']
        self.temporal_batch_edges = torch.tensor(
            [temporal_src_edges, temporal_dst_edges], dtype=torch.long)

        # Temporal Graph graph Layout
        self.graph_time_steps = graph_time_steps
        self.stride = stride

    def _get_scene_video_data(self, video_id, entity_id, mid_index, cache):
        orginal_entity_metadata = self.entity_data[video_id][entity_id]
        time_ent = orginal_entity_metadata[mid_index][1]
        context = self.get_speaker_context(video_id, entity_id, time_ent, self.context_size)

        video_data = []
        targets = []
        for ctx_entity in context:
            entity_metadata = self.entity_data[video_id][ctx_entity]
            ts_idx = self.search_ts_in_meta_data(entity_metadata, time_ent)
            target_ctx = int(entity_metadata[ts_idx][-1])

            clip_meta_data = cu.generate_clip_meta(entity_metadata, ts_idx, self.half_clip_length)
            video_data.append(io.load_v_clip_from_metadata_cache(clip_meta_data, self.video_root, cache))
            targets.append(target_ctx)

        if self.video_transform is not None:
            for vd_idx, vd in enumerate(video_data):
                tensor_vd = [self.video_transform(f) for f in vd]
                video_data[vd_idx] = tensor_vd

        if self.do_video_augment:
            video_data = [video_corner_crop(
                vd, self.crop_ratio) for vd in video_data]

        video_data = [torch.stack(vd, dim=1) for vd in video_data]
        return video_data, targets

    def _get_time_context(self, entity_data, target_index):
        all_ts = [ed[1] for ed in entity_data]
        center_ts = entity_data[target_index][1]
        center_ts_idx = all_ts.index(str(center_ts))

        half_time_steps = int(self.graph_time_steps/2)
        start = center_ts_idx-(half_time_steps*self.stride)
        end = center_ts_idx+((half_time_steps+1)*self.stride)
        selected_ts_idx = list(range(start, end, self.stride))

        selected_ts = []
        for i, idx in enumerate(selected_ts_idx):
            if idx < 0:
                idx = 0
            if idx >= len(all_ts):
                idx = len(all_ts)-1
            selected_ts.append(all_ts[idx])

        return selected_ts

    def __getitem__(self, index):
        video_id, entity_id = self.entity_list[index]
        target_entity_metadata = self.entity_data[video_id][entity_id]
        center_index = random.randint(0, len(target_entity_metadata)-1)
        time_context = self._get_time_context(
            target_entity_metadata, center_index)

        feature_set = None
        target_set = []

        all_ts = [ted[1] for ted in target_entity_metadata]
        nodes_per_graph = self.context_size+1
        cache = {}
        for graph_idx, tc in enumerate(time_context):
            target_index = all_ts.index(tc)

            # get av data
            video_data, target_v = self._get_scene_video_data(video_id, entity_id, target_index, cache)
            audio_data, target_a = self._get_audio_data(video_id, entity_id, target_index)

            # Fill targets
            target_set.append(target_a)
            for tv in target_v:
                target_set.append(tv)

            # Create now that we have the size
            if feature_set is None:
                feature_set = torch.zeros(nodes_per_graph * (self.graph_time_steps), video_data[0].size(0), video_data[0].size(1), video_data[0].size(2), video_data[0].size(3))

            # Fill in
            graph_offset = graph_idx*nodes_per_graph
            audio_data = torch.from_numpy(audio_data)
            feature_set[graph_offset, 0, 0, :audio_data.size(1), :audio_data.size(2)] = audio_data
            for i in range(self.context_size):
                feature_set[graph_offset + (i+1), ...] = video_data[i]

        return Data(x=feature_set, edge_index=(self.spatial_batch_edges, self.temporal_batch_edges), y=torch.tensor(target_set))
