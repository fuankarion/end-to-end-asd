import torch
import torch.nn as nn
import torch.nn.parameter

from functools import partial
from torch_geometric.nn import EdgeConv

from models.graph_layouts import generate_av_mask
from models.shared_2d import BasicBlock2D, conv1x1
from models.shared_3d import BasicBlock3D, Bottleneck3D, conv1x1x1, get_inplanes


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class LinearPathPreact(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LinearPathPreact, self).__init__()
        # Layer 1
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)

        # Layer 2
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Shared
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        x1 = self.fc1(x1)

        x2 = self.bn2(x1)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)

        return x2


class GraphTwoStreamResNet3D(nn.Module):
    def __init__(self, args_2d, args_3d):
        super().__init__()

        block_2d, layers_2d, zero_init_residual, groups_2d, width_per_group, replace_stride_with_dilation, norm_layer_2d = args_2d
        block_3d, layers_3d, block_inplanes_3d, n_input_channels, conv1_t_size, conv1_t_stride, no_max_pool, shortcut_type, widen_factor = args_3d

        # Global Args
        if norm_layer_2d is None:
            norm_layer_2d = nn.BatchNorm2d
        self._norm_layer_2d = norm_layer_2d

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        # Audio stream
        self.inplanes_2d = 64
        self.dilation_2d = 1
        self.groups_2d = groups_2d
        self.base_width = width_per_group

        self.audio_conv1 = nn.Conv2d(1, self.inplanes_2d, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.a_bn1 = norm_layer_2d(self.inplanes_2d)
        self.a_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a_layer1 = self._make_layer_2D(block_2d, 64, layers_2d[0])
        self.a_layer2 = self._make_layer_2D(block_2d, 128, layers_2d[1], stride=2,
                                            dilate=replace_stride_with_dilation[0])
        self.a_layer3 = self._make_layer_2D(block_2d, 256, layers_2d[2], stride=2,
                                            dilate=replace_stride_with_dilation[1])
        self.a_layer4 = self._make_layer_2D(block_2d, 512, layers_2d[3], stride=2,
                                            dilate=replace_stride_with_dilation[2])
        self.a_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_128_a = nn.Linear(512 * block_2d.expansion, 128)

        # Video Stream
        block_inplanes = [int(x * widen_factor) for x in block_inplanes_3d]
        self.in_planes_3d = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.v_conv1 = nn.Conv3d(n_input_channels, self.in_planes_3d,
                                 kernel_size=(conv1_t_size, 7, 7),
                                 stride=(conv1_t_stride, 2, 2),
                                 padding=(conv1_t_size // 2, 3, 3),
                                 bias=False)
        self.v_bn1 = nn.BatchNorm3d(self.in_planes_3d)
        self.v_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        self.v_layer1 = self._make_layer_3D(block_3d, block_inplanes[0], layers_3d[0],
                                            shortcut_type)
        self.v_layer2 = self._make_layer_3D(block_3d, block_inplanes[1], layers_3d[1],
                                            shortcut_type, stride=2)
        self.v_layer3 = self._make_layer_3D(block_3d, block_inplanes[2], layers_3d[2],
                                            shortcut_type, stride=2)
        self.v_layer4 = self._make_layer_3D(block_3d, block_inplanes[3], layers_3d[3],
                                            shortcut_type, stride=2)

        self.v_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_128_v = nn.Linear(512 * block_3d.expansion, 128)

        # Shared
        self.relu = nn.ReLU(inplace=True)

        # Dim reduction
        self.reduction_a = nn.Linear(512 * block_2d.expansion, 128)
        self.reduction_v = nn.Linear(512 * block_3d.expansion, 128)
        self.fc_aux_a = nn.Linear(128, 2)
        self.fc_aux_v = nn.Linear(128, 2)

        # Graph Net
        self.edge1 = EdgeConv(LinearPathPreact(128*2, 64))
        self.edge2 = EdgeConv(LinearPathPreact(64*2, 64))
        self.edge3 = EdgeConv(LinearPathPreact(64*2, 64))
        self.edge4 = EdgeConv(LinearPathPreact(64*2, 64))
        self.fc = nn.Linear(64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_2D(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer_2d
        downsample = None
        previous_dilation = self.dilation_2d
        if dilate:
            self.dilation_2d *= stride
            stride = 1
        if stride != 1 or self.inplanes_2d != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_2d, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_2d, planes, stride, downsample, self.groups_2d,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes_2d = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes_2d, planes, groups=self.groups_2d,
                                base_width=self.base_width, dilation=self.dilation_2d,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer_3D(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes_3d != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes_3d, planes
                              * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(in_planes=self.in_planes_3d, planes=planes,
                            stride=stride, downsample=downsample))
        self.in_planes_3d = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes_3d, planes))

        return nn.Sequential(*layers)

    def forward_audio(self, a, audio_size):
        a = torch.unsqueeze(a[:, 0, 0, :audio_size[1], :audio_size[2]], dim=1)

        a = self.audio_conv1(a)
        a = self.a_bn1(a)
        a = self.relu(a)
        a = self.a_maxpool(a)

        a = self.a_layer1(a)
        a = self.a_layer2(a)
        a = self.a_layer3(a)
        a = self.a_layer4(a)

        a = self.a_avgpool(a)
        a = a.reshape(a.size(0), -1)
        return a

    def forward_video(self, v):
        v = self.v_conv1(v)
        v = self.v_bn1(v)
        v = self.relu(v)
        if not self.no_max_pool:
            v = self.v_maxpool(v)

        v = self.v_layer1(v)
        v = self.v_layer2(v)
        v = self.v_layer3(v)
        v = self.v_layer4(v)

        v = self.v_avgpool(v)
        v = v.reshape(v.size(0), -1)
        return v

    def forward(self, data, ctx_size, audio_size):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        # indexing masks
        audio_mask, video_mask = generate_av_mask(ctx_size, x.size(0))

        # Initial Conv. forward
        audio_feats = self.forward_audio(x[audio_mask], audio_size)
        video_feats = self.forward_video(x[video_mask])

        # Dim Reduction
        audio_feats = self.relu(self.reduction_a(audio_feats))
        video_feats = self.relu(self.reduction_v(video_feats))

        # Rebuild interleaved tensor
        graph_feats = torch.zeros(
            (x.size(0), 128), device=audio_feats.get_device(), dtype=audio_feats.dtype)
        graph_feats[audio_mask] = audio_feats
        graph_feats[video_mask] = video_feats

        # Aux supervision
        audio_out = self.fc_aux_a(graph_feats[audio_mask])
        video_out = self.fc_aux_v(graph_feats[video_mask])

        # Graph Stream
        graph_feats = self.edge1(graph_feats, edge_index)
        graph_feats = self.edge2(graph_feats, edge_index)
        graph_feats = self.edge3(graph_feats, edge_index)
        graph_feats = self.edge4(graph_feats, edge_index)
        return self.fc(graph_feats), audio_out, video_out


class GraphTwoStreamResNet3DTwoGraphs4LVLRes(GraphTwoStreamResNet3D):
    def __init__(self, args_2d, args_3d, filter_size):
        super().__init__(args_2d, args_3d)
        self.edge_spatial_1 = EdgeConv(LinearPathPreact(128*2, filter_size))
        self.edge_spatial_2 = EdgeConv(LinearPathPreact(filter_size*2, filter_size))
        self.edge_spatial_3 = EdgeConv(LinearPathPreact(filter_size*2, filter_size))
        self.edge_spatial_4 = EdgeConv(LinearPathPreact(filter_size*2, filter_size))

        self.edge_temporal_1 = EdgeConv(LinearPathPreact(filter_size*2, filter_size))
        self.edge_temporal_2 = EdgeConv(LinearPathPreact(filter_size*2, filter_size))
        self.edge_temporal_3 = EdgeConv(LinearPathPreact(filter_size*2, filter_size))
        self.edge_temporal_4 = EdgeConv(LinearPathPreact(filter_size*2, filter_size))

        self.fc = nn.Linear(filter_size, 2)

        # IS this necessary?
        self.edge1 = None
        self.edge2 = None
        self.edge3 = None
        self.edge4 = None

    def forward(self, data, ctx_size, audio_size):
        x, joint_edge_index, _ = data.x, data.edge_index, data.batch
        spatial_edge_index = joint_edge_index[0]
        temporal_edge_index = joint_edge_index[1]

        # indexing masks
        audio_mask, video_mask = generate_av_mask(ctx_size, x.size(0))

        # Initial Conv. forward
        audio_feats = self.forward_audio(x[audio_mask], audio_size)
        video_feats = self.forward_video(x[video_mask])

        # Dim Reduction
        audio_feats = self.relu(self.reduction_a(audio_feats))
        video_feats = self.relu(self.reduction_v(video_feats))

        # Rebuild interleaved tensor
        graph_feats = torch.zeros((x.size(0), 128), device=audio_feats.get_device(), dtype=audio_feats.dtype)
        graph_feats[audio_mask] = audio_feats
        graph_feats[video_mask] = video_feats

        # Aux supervision
        audio_out = self.fc_aux_a(graph_feats[audio_mask])
        video_out = self.fc_aux_v(graph_feats[video_mask])

        # Spatial Stream
        graph_feats_1s = self.edge_spatial_1(graph_feats, spatial_edge_index)
        graph_feats_1st = self.edge_temporal_1(graph_feats_1s, temporal_edge_index)

        graph_feats_2s = self.edge_spatial_2(graph_feats_1st, spatial_edge_index)
        graph_feats_2st = self.edge_temporal_2(graph_feats_2s, temporal_edge_index)
        graph_feats_2st = graph_feats_2st + graph_feats_1st

        graph_feats_3s = self.edge_spatial_3(graph_feats_2st, spatial_edge_index)
        graph_feats_3st = self.edge_temporal_3(graph_feats_3s, temporal_edge_index)
        graph_feats_3st = graph_feats_3st + graph_feats_2st

        graph_feats_4s = self.edge_spatial_4(graph_feats_3st, spatial_edge_index)
        graph_feats_4st = self.edge_temporal_4(graph_feats_4s, temporal_edge_index)
        graph_feats_4st = graph_feats_4st + graph_feats_3st

        return self.fc(graph_feats_4st), audio_out, video_out


def _load_video_weights_into_model(model, ws_file):
    resnet_state_dict = torch.load(ws_file)['state_dict']

    own_state = model.state_dict()
    for name, param in resnet_state_dict.items():
        if 'v_'+name in own_state:
            own_state['v_'+name].copy_(param)
        else:
            print('No video assignation for ', name)

    print('loaded video ws')
    return 


def _load_audio_weights_into_model(model, arch2d, progress):
    resnet_state_dict = load_state_dict_from_url(
        model_urls[arch2d], progress=progress)

    own_state = model.state_dict()
    for name, param in resnet_state_dict.items():
        if 'a_'+name in own_state:
            own_state['a_'+name].copy_(param)
        else:
            print('No audio assignation for ', name)

    # Audio initial Ws
    conv1_weights = resnet_state_dict['conv1.weight']
    avgWs = torch.mean(conv1_weights, dim=1, keepdim=True)
    own_state['audio_conv1.weight'].copy_(avgWs)

    print('loaded audio ws')
    return 


def R3D18_4lvlGCN(pretrained_weigths, filter_size=128):
    args_2d = BasicBlock2D, [2, 2, 2, 2], False, 1, 64, None, None
    args_3d = BasicBlock3D, [2, 2, 2, 2], get_inplanes(), 3, 7, 1, False, 'B', 1.0
    model = GraphTwoStreamResNet3DTwoGraphs4LVLRes(args_2d, args_3d, filter_size)

    _load_audio_weights_into_model(model, 'resnet18', True)
    _load_video_weights_into_model(model, pretrained_weigths)

    return model


def R3D50_4lvlGCN(pretrained_weigths, filter_size=128):
    args_2d = BasicBlock2D, [2, 2, 2, 2], False, 1, 64, None, None
    args_3d = Bottleneck3D, [3, 4, 6, 3], get_inplanes(), 3, 7, 1, False, 'B', 1.0
    model = GraphTwoStreamResNet3DTwoGraphs4LVLRes(args_2d, args_3d, filter_size)

    _load_audio_weights_into_model(model, 'resnet18', True)
    _load_video_weights_into_model(model, pretrained_weigths)

    return model
