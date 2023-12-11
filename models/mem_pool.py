from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Conv2d, KLDivLoss, Linear, Parameter

from torch_geometric.utils import to_dense_batch

EPS = 1e-15


class Semantic_Component_Level_Memory(torch.nn.Module):
    

    def __init__(self, in_channels: int, out_channels: int, heads: int,
                 num_clusters: int, tau: float = 1.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_clusters = num_clusters
        self.tau = tau

        #这个是以前的k的定义
        #self.k = Parameter(torch.Tensor(heads, num_clusters, in_channels))

        #因为spg中没有真实的填充类别的特征，这里搞个全0做。
        # cluster_features=torch.load("/home/wsy/sketch/code_4/explain/SketchXAI-main_28/yu_xun_lian_features/clusters_features_3.pth")['clusters_features'].cuda()
        # padding=torch.randn((1,3,768)).cuda()
        # cluster_features=torch.cat([cluster_features,padding],dim=0)

        embedding_fn=torch.nn.Embedding(num_clusters*heads,768)
        cluster_index=torch.arange(start=0,end=261)

        cluster_features=embedding_fn(cluster_index).cuda().view(num_clusters,heads,-1)


        cluster_features=cluster_features.transpose(1,0).contiguous().cpu()
        
        self.k=Parameter(torch.Tensor(cluster_features),requires_grad=True)
        
        self.conv = Conv2d(heads, 1, kernel_size=1, padding=0, bias=False)
        
        #这里把原本的多头融合的卷积改成一个maxpool
        self.pool=torch.nn.MaxPool1d(heads)

        self.lin = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.uniform_(self.k.data, -1., 1.)
        self.conv.reset_parameters()
        self.lin.reset_parameters()

    @staticmethod
    def kl_loss(S: Tensor) -> Tensor:
        r"""The additional KL divergence-based loss

        .. math::
            P_{i,j} &= \frac{S_{i,j}^2 / \sum_{n=1}^N S_{n,j}}{\sum_{k=1}^K
            S_{i,k}^2 / \sum_{n=1}^N S_{n,k}}

            \mathcal{L}_{\textrm{KL}} &= \textrm{KLDiv}(\mathbf{P} \Vert
            \mathbf{S})
        """
        S_2 = S**2
        P = S_2 / S.sum(dim=1, keepdim=True)
        denom = P.sum(dim=2, keepdim=True)
        denom[S.sum(dim=2, keepdim=True) == 0.0] = 1.0
        P /= denom

        loss = KLDivLoss(reduction='batchmean', log_target=False)
        return loss(S.clamp(EPS).log(), P.clamp(EPS))

    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x (Tensor): Dense or sparse node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}` or
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`,
                respectively.
            batch (LongTensor, optional): Batch vector :math:`\mathbf{b} \in
                {\{ 0, \ldots, B-1\}}^N`, which assigns each node to a
                specific example.
                This argument should be just to separate graphs when using
                sparse node features. (default: :obj:`None`)
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}`, which
                indicates valid nodes for each graph when using dense node
                features. (default: :obj:`None`)
        """
        if x.dim() <= 2:
            x, mask = to_dense_batch(x, batch)
        elif mask is None:
            mask = x.new_ones((x.size(0), x.size(1)), dtype=torch.bool)

        (B, N, _), H, K = x.size(), self.heads, self.num_clusters


        #计算每个聚类中心和边距离
        dist = torch.cdist(self.k.view(H * K, -1), x.view(B * N, -1), p=2)**2
        dist = dist.view(H, K, B, N).permute(2, 3, 0, 1)  # [B, N, H, K]

        

        #每个边到每个中心的距离除以均值
        dist = (dist / dist.mean(dim=-1, keepdim=True))
        dist = dist.permute(0,1,3,2).view(B*N,K,H) #[B,N,K,H]
        #距离取倒数
        dist = ( dist / self.tau).pow(-(self.tau + 1.0) / 2.0)

        
        #多头进行一个最大池化
        
        S = self.pool(dist).squeeze(-1).view(B,N,K)

        #这里先成了个系数5，我看了下训练集最后大概会在softmax前是5，15，结果出来是真值上是99，测试集中是5，8--10到不了很高，softmax后是真值是50左右概率。
        #上面的数据都是看的都是个例。
        S = S * mask.view(B, N, 1)*5

        S=S.softmax(dim=-1)

        #先把key的特征复制成B*K*3*768
        #通过距离得到每个头的索引，最终变成B*K*768

        #先把key的特征复制成B*K*3*768
        k=self.k.permute(1,0,2)
        k=torch.repeat_interleave(k.unsqueeze(0),repeats=B,dim=0).view(-1,3,768)

        #下面是找到每个头的索引
        #距离为B*N*K*768
        #先转成B*3*K*N,通过max得到B*3*K,这意思是先看sketch中每个组件的最大概率（通过每个N），这是需要的值还是  组件概率
        #然后转成B*K*3，对3维度做一个max，得到最大的头的  索引  ，得到B*K

        #距离这里最后可以再乘上一个mask矩阵
        S1=dist.view(B,N,K,H)
        S1=S1.permute(0,3,2,1)  #B H K N
        S1,_=torch.topk(S1,k=1,dim=3) #B H K 1
        S1=S1.squeeze(-1)
        
        S1=S1.permute(0,2,1) #B K H
        _,S1=torch.topk(S1,k=1,dim=2)
        S1=S1.squeeze(-1).view(-1)

        S1=S1.unsqueeze(-1)
        S1=S1.repeat(1,768)
        S1=S1.unsqueeze(1)
        
        #根据索引和k得到最大的头
        #即得到B*K*768
        final_k=torch.gather(k,1,S1).squeeze(1).view(B,K,768)

        #下面的到根据每概率矩阵和cluster的特征，得到每个笔画基于cluster的特征

        stroke_cluster=torch.bmm(S,final_k)
        f=stroke_cluster

        # _,S_max=torch.topk(S,k=1,dim=-1)
        # S_max=torch.repeat_interleave(S_max,repeats=768,dim=-1)
        
        # f=torch.mul((1-S_max),x)+torch.mul(S_max,stroke_cluster)

        f_padding=torch.zeros((f.size()[0],43-f.size()[1],768)).cuda()
        
        f=torch.cat([f,f_padding],dim=1)
        x1 = self.lin(S.transpose(1, 2) @ x)
        
        k=self.k.permute(1,0,2).contiguous().view(-1,768)

        
        return f, S, k

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'num_clusters={self.num_clusters})')



