import math


def get_spatial_connection_pattern(ctx_size, num_graphs):
    cp = {}
    cp['src'] = []
    cp['dst'] = []

    #  Self connections
    for g in range(num_graphs):
        graph_offset = g*(ctx_size+1)
        for s in range(ctx_size+1):
            cp['src'].append(graph_offset + s)
            cp['dst'].append(graph_offset + s)

    # Spatial AV connections
    for g in range(num_graphs):
        graph_offset = g*(ctx_size+1)
        for s in range(1, ctx_size+1):
            cp['src'].append(graph_offset + 0)
            cp['dst'].append(graph_offset + s)

            cp['src'].append(graph_offset + s)
            cp['dst'].append(graph_offset + 0)

    # Spatial VV connections
    for g in range(num_graphs):
        graph_offset = g*(ctx_size+1)

        for s in range(1, ctx_size+1):
            for d in range(1, ctx_size+1):
                if d != s:
                    cp['src'].append(graph_offset + s)
                    cp['dst'].append(graph_offset + d)

    return cp


def get_temporal_connection_pattern(ctx_size, num_graphs):
    cp = {}
    cp['src'] = []
    cp['dst'] = []

    #  Self connections
    for g in range(num_graphs):
        graph_offset = g*(ctx_size+1)
        for s in range(ctx_size+1):
            cp['src'].append(graph_offset + s)
            cp['dst'].append(graph_offset + s)

    # Temporal VV connections
    for g in range(num_graphs):
        graph_offset = g*(ctx_size+1)
        for s in range(0, ctx_size+1):

            if g > 0:
                left_graph_offset = (g-1)*(ctx_size+1)
                cp['src'].append(graph_offset + s)
                cp['dst'].append(left_graph_offset + s)

            if g < num_graphs - 1:
                right_graph_offset = (g+1)*(ctx_size+1)
                cp['src'].append(graph_offset + s)
                cp['dst'].append(right_graph_offset + s)

    return cp


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


def generate_temporal_video_mask(ctx_size, total_len):
    stride = ctx_size + 1
    video_mask = [i for i in range(1, total_len, stride)]
    return video_mask


def generate_temporal_video_center_mask(ctx_size, total_len, time_len):
    stride = ctx_size + 1
    video_mask = [i + stride*math.floor(time_len/2)
                  for i in range(1, total_len, stride*time_len)]
    return video_mask
