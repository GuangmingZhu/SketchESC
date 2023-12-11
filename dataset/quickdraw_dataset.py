from __future__ import division
from __future__ import print_function

import os.path as osp
import pickle
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data as pygData
import ndjson

def get_graph_data(stroke_num):
    graph_data=[]

    if stroke_num>1:
        for i in range(stroke_num-1):
            
            graph_data.append([i,i+1])
            graph_data.append([i+1,i])
    else:
        graph_data.append([0,0])
        graph_data.append([0,0])

    graph_data = np.array(graph_data).transpose()
    return graph_data

class QuickDrawDataset(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.mode = mode
        self.data = None

        with open(osp.join(root_dir, "SPG_"+mode+".ndjson"), 'r') as fh:
            
            self.content=ndjson.load(fh)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        
        # drawing=np.array(self.content[idx]["drawing"]).astype(np.float32)
        category=torch.tensor(self.content[idx]["recog_label"])
        seg_label=torch.tensor(np.array(self.content[idx]["seg_label"]))
        seg_label1=torch.tensor(np.array(self.content[idx]["seg_label1"]))

        points_offset=torch.tensor(np.array(self.content[idx]['points_offsets']))
        position_list=torch.tensor(np.array(self.content[idx]['position_list']))
        stroke_number=torch.tensor(self.content[idx]['stroke_num'])
        stroke_mask=torch.tensor(np.array(self.content[idx]['stroke_mask']))
        sketch_stroke_num=torch.tensor(self.content[idx]['sketch_stroke_num'])

        sketch_components_num=torch.tensor(self.content[idx]['sketch_components_num'])
        key_id=torch.tensor(int(self.content[idx]['key_id'])).long()

        edge_index=get_graph_data(self.content[idx]['sketch_stroke_num'])
        graph_data=pygData(x=torch.zeros((self.content[idx]['sketch_stroke_num'],2)),edge_index=torch.LongTensor(edge_index))
        
        seg_label2=torch.zeros((87))
        for i in range(sketch_components_num):
            seg_label2[seg_label1[i]]=1


        sample = {'points_offset':points_offset, 'category': category, 'seg_label': seg_label, 'seg_label1': seg_label1,'position_list':position_list,'stroke_number':stroke_number,'stroke_mask':stroke_mask,'sketch_stroke_num':sketch_stroke_num,'graph_data':graph_data,'key_id':key_id,'seg_label2':seg_label2}
        
         
        return sample

    def num_categories(self):
        return 20
