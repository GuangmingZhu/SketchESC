import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
from .dynamic import DilatedKnnGraph

def cal_edge_attr(edge_index, pos):
    row = edge_index[0]
    col = edge_index[1]
    offset = pos[col] - pos[row]
    dist = torch.norm(offset, p=2, dim=-1).view(-1, 1)
    theta = torch.atan2(offset[:,1], offset[:,0]).view(-1, 1)
    edge_attr = torch.cat([offset, dist, theta], dim=-1)
    return edge_attr

def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer

def norm_layer(norm_type, nc):
    """
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer

class MLPLinear(nn.Sequential):
    def __init__(self, channels, act_type='relu', norm_type='batch', bias=True):
        m = []
        
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))
            if norm_type and norm_type != 'None':
                
                m.append(norm_layer(norm_type, channels[i]))
                
            if act_type:
                m.append(act_layer(act_type))
        super(MLPLinear, self).__init__(*m)

class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.gcn_type = 'edge'
        if self.gcn_type == 'edge':
            self.gconv = tg.nn.EdgeConv(
                nn=MLPLinear(
                    channels=[in_channels*2, out_channels],
                    act_type='relu', 
                    norm_type=None
                ),
                aggr='max'
            )
        elif self.gcn_type == 'ecc':
            self.gconv = tg.nn.NNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                nn=nn.Linear(4, in_channels*out_channels, bias=False),
                aggr='mean',
                root_weight=False,
                bias=True
            )            
        else:
            raise NotImplementedError('conv_type {} is not implemented. Please check.\n'.format('graph_conv'))

    def forward(self, x, edge_index, data=None):
        """
        x: (BxN) x F
        """
        
        if self.gcn_type == 'ecc':
            return self.gconv(x, edge_index, data['edge_attr'])
        else:
            return self.gconv(x, edge_index)

class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, dilation, knn_type='matrix',kernel_size=2):
        super(DynConv, self).__init__(in_channels, out_channels)
        #这里把kernel_size先设置为2，在做扩散的时候。
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(2, dilation)
        self.mixedge = True

    def forward(self, x, edge_index, batch,sketch_stroke_num):
        """
        x: (BxN) x F
        """
        dyn_edge_index = self.dilated_knn_graph(x, batch,sketch_stroke_num)
        if self.mixedge:
            dyn_edge_index = torch.unique(torch.cat([edge_index, dyn_edge_index], dim=1), dim=1)
        
        # TODO: calculate edge_attr use pos
        # if self.gcn_type == 'ecc':
        #     dyn_edge_attr = cal_edge_attr(dyn_edge_index, data['pos'])
        # else:
        #     dyn_edge_attr = None
        dyn_edge_attr=None
        
        return super(DynConv, self).forward(x, dyn_edge_index, {'edge_attr':dyn_edge_attr})