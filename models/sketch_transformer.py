import numpy as np
import torch
from torch import nn
from transformers import ViTForImageClassification, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.vit.modeling_vit import ViTEmbeddings
from .graph_model import DynConv
from .pool import Pool
import  torch_geometric.nn as pyg_nn

from .mem_pool import Semantic_Component_Level_Memory

class StrokeEmbeddings(nn.Module):
    """
    Construct the CLS token, stroke, order and position embeddings.

    """

    def __init__(self,  opt, use_mask_token: bool = True):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        lstm_hidden_size = int(768 / 2)
        if opt['shape_extractor'] == 'lstm':
            self.stroke_embeddings = nn.LSTM(4, lstm_hidden_size, num_layers=opt['shape_extractor_layer'], batch_first=True, bidirectional=True)
        elif opt['shape_extractor'] == 'gru':
            self.stroke_embeddings = nn.GRU(4, lstm_hidden_size, num_layers=opt['shape_extractor_layer'], batch_first=True, bidirectional=True)

        num_patches = opt['max_stroke']
        self.num_patches = num_patches
        self.batch_size = opt['bs']
        self.order_embeddings = nn.Embedding(num_patches, 768)
        self.order = torch.arange(0, self.num_patches)
        if use_mask_token:
            mask_token = torch.zeros(1, 1, 768)
            self.mask_tokens = mask_token.expand(self.batch_size, self.num_patches + 1, -1)

        self.location_embeddings = nn.Linear(2, 768)
        self.shape_func = opt['shape_emb']
        
        self.opt = opt

    # N x 768 -> bs x 196 x 768  N x 2 -> bs x 196 x 2
    def reconstruct_batch(self, embeddings, position_values, stroke_number):
        devices = embeddings.get_device()
        #batch_size*196*768

        batch_embeddings = torch.zeros(len(stroke_number), self.num_patches, 768, device=devices)
        batch_positions = torch.zeros(len(stroke_number), self.num_patches, 2, device=devices)
        #每个sketch有多少个笔画
        strokes = np.asarray([stroke.size for stroke in stroke_number])

        for index_sketch in range(strokes.size):
            sketch_strokes = strokes[index_sketch]
            start = np.sum(strokes[:index_sketch])
            batch_embeddings[index_sketch, :sketch_strokes, :] = embeddings[start:start + sketch_strokes, :]
            batch_positions[index_sketch, :sketch_strokes, :] = position_values[start:start + sketch_strokes, :]
        return batch_embeddings, batch_positions

    # N x stroke_length x 4 -> N x 768
    def lstm_out(self, embed, text_length):
        
        #把当前batch里每个笔画的长度放到一个维度里（不区分是哪个sketch的）
        stroke_length_order = np.hstack(text_length)
        
        length_tensor = torch.from_numpy(stroke_length_order).to(embed.get_device())
        _, idx_sort = torch.sort(length_tensor, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        #把每个笔画根据壁画长度排序后，得到每个笔画的点序列和每个笔画实际的长度序列
        embed_sort = embed.index_select(0, idx_sort).float()
        length_list = length_tensor[idx_sort]
        
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )
        
        sort_out, _ = self.stroke_embeddings(pack)
        sort_out = nn.utils.rnn.pad_packed_sequence(sort_out, batch_first=True)
        sort_out = sort_out[0]

        #output_size:[stroke_num,max_stroke_point_num,embed_num]
        output = sort_out.index_select(0, idx_unsort)
        
        

        if self.shape_func == 'sum':
            output = torch.sum(output, dim=1)
        elif self.shape_func == 'mean':
            output = torch.mean(output, dim=1)
        
        #output_size:[stroke_num,embed_num]
        
        return output

    def forward(self, points_values, position_values, stroke_number, bool_masked_pos=None):
        devices = points_values.get_device()
        shape_emb = self.lstm_out(points_values, stroke_number)  # N x 768
        shape_emb, new_position_values = self.reconstruct_batch(shape_emb, position_values, stroke_number)  # bs x 196 x 768  bs x 196 x 2
        
        # add order embeddings
        order_emb = self.order_embeddings(self.order.to(devices)).unsqueeze(0)  # 1 x 196 x 768

        # add location embeddings
        location_emb = self.location_embeddings(new_position_values)  # bs x 196 x 768

        embeddings = shape_emb + order_emb + location_emb
        
        
        # add the [CLS] token to the embedded patch tokens
        #cls_tokens = self.cls_token.expand(self.batch_size, -1, -1).to(devices)  # bs x 1 x 768
        #cls_tokens = self.cls_token.expand(len(stroke_number), -1, -1).to(devices)
        #embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # bs x 197 x 768

        # if bool_masked_pos is not None:
            
        #     mask = bool_masked_pos.unsqueeze(-1).type_as(self.mask_tokens).to(devices)
        #     embeddings = embeddings * (1.0 - mask) + self.mask_tokens.to(devices) * mask

        
        return embeddings


class SketchViT(ViTModel):
    def __init__(self,config,  opt,labels_number=20, add_pooling_layer=True, use_mask_token: bool = True):
        super().__init__(config,add_pooling_layer)
        self.embeddings = StrokeEmbeddings( opt, use_mask_token=use_mask_token)
        
        self.fc=nn.Linear(768,20)
        self.seg_fc=nn.Sequential(
            nn.Linear(768,768),
            nn.ReLU(),
            nn.Linear(768,87)
        )
        self.seg_fc2=nn.Sequential(
            nn.Linear(768,768),
            nn.ReLU(),
            nn.Linear(768,87)
        )

        self.encoder_seg_fc=nn.Linear(768,87)

        self.channels=768
        #动态卷积，kernel_size表示动态卷积看到的邻接范围，默认为2，最终看到的邻接范围为dilation*kernel_size
        self.graph_conv1=DynConv(in_channels= self.channels, out_channels= self.channels, dilation=1, knn_type='matrix')
        self.graph_conv2=DynConv(in_channels= self.channels, out_channels= self.channels, dilation=2, knn_type='matrix')

        #这个是pooling的，先不用
        self.graph_conv3=DynConv(in_channels= self.channels, out_channels= 8, dilation=1, knn_type='matrix',kernel_size=1)

        self.pool=Semantic_Component_Level_Memory(in_channels=768,out_channels=768,heads=3,num_clusters=87)
        #self.pool=mem_pool1(in_channels=768,out_channels=768,heads=3,num_clusters=86)

        #transformer的cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

        #从batch*87*768->batch*87*2，这里的2表示2分类任务，即当前的组件在还是不在
        self.linear1=nn.Linear(in_features=768,out_features=2)

        #设置新的转换矩阵，使得S*X*W得到结果。这里S为原本的抽取8个组件后的S，且表示为转置，
        #改了memPool的输出，使其输出feature，S，和未与SW相乘的X,这样就可以将X与修改后的S相乘了
        #没用到！！！
        self.linear2=nn.Linear(in_features=768,out_features=768)

        #transformer的cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

    def forward(
            self,
            point_values=None,
            position_values=None,
            stroke_number=None,
            bool_masked_pos=None,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            interpolate_pos_encoding=None,
            return_dict=None,
            graph_data=None,
            sketch_stroke_num=None
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        #b*43*768,做完了lstm
        embedding_output = self.embeddings(point_values, position_values, stroke_number, bool_masked_pos=bool_masked_pos)


        #开始进行gnn部分
        #先将结点从b*43*768抽取成n*768，n为batch中的node总数
        nodes_features=torch.zeros((1,768)).cuda()
        for i in range(embedding_output.size()[0]):
            nodes_features=torch.cat([nodes_features,embedding_output[i][:len(stroke_number[i])]])
        nodes_features=nodes_features[1:]
        
        #进行图卷积
        nodes_features_1=self.graph_conv1(nodes_features,graph_data.edge_index,graph_data.batch,sketch_stroke_num)
        nodes_features=nodes_features+nodes_features_1
        nodes_features_2=self.graph_conv1(nodes_features,graph_data.edge_index,graph_data.batch,sketch_stroke_num)
        nodes_features=nodes_features+nodes_features_2


        #进行mempooling
        nodes_features1,mapping, clusters_features=self.pool(nodes_features,graph_data.batch)

        cls_tokens = self.cls_token.expand(nodes_features1.size()[0], -1, -1).cuda()

        
        nodes_features_transformer = torch.cat((cls_tokens, nodes_features1), dim=1)

        encoder_outputs = self.encoder(
            nodes_features_transformer,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        

        logits = self.fc(sequence_output[:, 0, :])
        encoder_seg_outs= self.encoder_seg_fc(sequence_output[:,1:,:])

        cluster_seg_outs=self.seg_fc(clusters_features)
        
        graph_outputs=self.seg_fc2(nodes_features)

        

        #返回分类结果，映射矩阵，每个笔画的分类结果
        return logits,mapping,encoder_seg_outs,cluster_seg_outs,graph_outputs, BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_order_embedding(self):
        return self.embeddings.order_embeddings.weight
    

class ViTForSketchClassification(ViTForImageClassification):
    def __init__(self, config, opt, labels_number=20, use_mask_token: bool = True):
        super().__init__(config)
        self.vit = SketchViT(config, opt, labels_number, add_pooling_layer=False, use_mask_token=use_mask_token)

    def forward(
            self,
            point_values=None,
            position_values=None,
            stroke_number=None,
            bool_masked_pos=None,
            graph_data=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            interpolate_pos_encoding=None,
            return_dict=None,
            sketch_stroke_num=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        logits,seg_outs,encoder_seg_outs,cluster_seg_outs,graph_seg_outs, outputs = self.vit(
            point_values,
            position_values,
            stroke_number,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            graph_data=graph_data,
            sketch_stroke_num=sketch_stroke_num
        )

        return logits,seg_outs,encoder_seg_outs,cluster_seg_outs, graph_seg_outs, outputs.last_hidden_state, outputs.attentions


