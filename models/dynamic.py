import torch
from torch import nn
from torch_cluster import knn_graph


class Dilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(Dilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index, batch=None):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index.view(2, -1, num)
                edge_index = edge_index[:, :, randnum]
                return edge_index.view(2, -1)
            else:
                edge_index = edge_index[:, ::self.dilation]
        else:
            edge_index = edge_index[:, ::self.dilation]
        return edge_index
    
    def __repr__(self):
        return self.__class__.__name__ + ' (k=' + str(self.k) + ', d=' + str(self.dilation) + ', e=' + str(self.epsilon) + ')'


class DilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=True, epsilon=0.1, knn_type='matrix'):
        super(DilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = Dilated(k, dilation, stochastic, epsilon)
        if knn_type == 'matrix':
            self.knn = knn_graph_matrix
        else:
            self.knn = knn_graph

    def forward(self, x, batch, sketch_stroke_num):
        edge_index = self.knn(x, self.k * self.dilation, batch,sketch_stroke_num)
        return self._dilated(edge_index, batch)


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def knn_matrix(x, k=16, batch=None,sketch_stroke_num=None):
    """Get KNN based on the pairwise distance.
    Args:
        pairwise distance: (num_points, num_points)
        k: int
    Returns:
        nearest neighbors: (num_points*k ,1) (num_points, k)
    """
    

    #原本的knn面对的是每个草图的节点数量一样的情况，我们的每个草图结点数量不一样，所以得根据batch，对每个单独进行处理，得到每个草图的边的情况。

    nn_idxs=torch.zeros((1,1)).cuda()
    center_idxs=torch.zeros((1,1)).cuda()
    for i in range(len(sketch_stroke_num)):
        stroke_start_idx=torch.tensor(0)
        if i==0:
            x1=x[:sketch_stroke_num[i]]
        else:
            x1=x[torch.sum(sketch_stroke_num[:i]):torch.sum(sketch_stroke_num[:i+1])]
            stroke_start_idx=torch.sum(sketch_stroke_num[:i])

        #x1:n*f,n为当前草图的笔画数量
        x1=x1.unsqueeze(0)
        neg_adj = -pairwise_distance(x1)
        _, nn_idx = torch.topk(neg_adj, k=k)
        del neg_adj

        n_points = x1.shape[1]
        if x.is_cuda:
            start_idx = stroke_start_idx.cuda()
        nn_idx += start_idx
        del stroke_start_idx

        if x1.is_cuda:
            torch.cuda.empty_cache()

        nn_idx = nn_idx.view(1, -1)
        center_idx = torch.arange(0, n_points).repeat(k, 1).transpose(1, 0).contiguous().view(1, -1).cuda()
        if x1.is_cuda:
            center_idx = center_idx.cuda()

        center_idx=center_idx+start_idx

        nn_idxs=torch.cat([nn_idxs,nn_idx],dim=1)
        center_idxs=torch.cat([center_idxs,center_idx],dim=1)
    nn_idxs=nn_idxs[:,1:].long()
    center_idxs=center_idxs[:,1:].long()
    
    return nn_idxs, center_idxs


def knn_graph_matrix(x, k=16, batch=None,sketch_stroke_num=None):
    """Construct edge feature for each point
    Args:
        x: (num_points, num_dims)
        batch: (num_points, )
        k: int
    Returns:
        edge_index: (2, num_points*k)
    """
      
    nn_idx, center_idx = knn_matrix(x, k, batch,sketch_stroke_num)
    return torch.cat((nn_idx, center_idx), dim=0)

