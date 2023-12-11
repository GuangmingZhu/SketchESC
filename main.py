import datetime
import os
import random

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import numpy as np
import torch
# import torch.distributed as dist
import torch.nn as nn
import torch.optim
import torch.optim as optim
import torch.utils.data
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch_geometric.data import DataLoader 
# from torch.utils.data import DataLoader

import opts
from dataset.quickdraw_dataset import QuickDrawDataset
from models.sketch_transformer import ViTForSketchClassification
from tqdm import tqdm
import time
import torch.multiprocessing 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

torch.multiprocessing.set_sharing_strategy('file_system')

t=time.localtime()

ckpt_folder = 'ckpt/'+str(t.tm_mon)+"-"+str(t.tm_mday)+"-"+str(t.tm_hour)+"-"+str(t.tm_min)+"-"+str(t.tm_sec)+"/"
log_folder = 'log/'+str(t.tm_mon)+"-"+str(t.tm_mday)+"-"+str(t.tm_hour)+"-"+str(t.tm_min)+"-"+str(t.tm_sec)+"/"
log_file = 'acc.txt'
train_log_file='train_acc.txt'

#max_stroke = 196
max_stroke = 43
cluster=9
#填充的笔画的label的真值设置为86
additional_seg_label=86

the_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def check_all():
    check_folder(ckpt_folder)
    check_folder(log_folder)
    check_log(log_file)

def check_folder(folder):
    folder = home + folder
    print('folder ' + folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

def check_log(file):
    file = home + log_folder + file
    print(file)
    if not os.path.exists(file):
        os.mknod(file)

def get_optim(model, lr, weight_decay):
    torch_optimizer = optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    return torch_optimizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, url):
    print('save_checkpoint: ' + str(url))
    torch.save(model.state_dict(),home + ckpt_folder + url)

def test(loader, model, devices, opt):
    with torch.no_grad():
        model.eval()
        running_corrects = 0
        seg_running_corrects=0
        seg_num=0
        loss_sum=0

        encoder_seg_corrects=0
        encoder_seg_num=0
        # loader.sampler.set_epoch(0)
        for it, data_batch in enumerate(loader):
            
            #b*43*256
            input_strokes = data_batch['points_offset'].to(devices)
            #b*43*2
            input_positions = data_batch['position_list'].to(devices)
            #b*43
            stroke_point_number = data_batch['stroke_number']
            #b*43
            stroke_mask = data_batch['stroke_mask'].to(devices) if opt['mask'] else None
            #b
            sketch_stroke_num=data_batch['sketch_stroke_num'].to(devices)

            graph_data=data_batch['graph_data'].to(devices)

            encoder_seg_label=data_batch['seg_label2'].to(devices)
            
            seg_label=data_batch['seg_label'].to(devices)

            #把每个笔画属于哪个sketch，每个笔画包含多少个point梳理一下，去掉填充，转成nparray
            real_stroke_point_number=[]
            for i in range(stroke_point_number.size()[0]):
                real_stroke_point_number.append(stroke_point_number[i][:sketch_stroke_num[i]].numpy())
            
            
            #把每个笔画和每个笔画的初始位置去掉填充
            #n*256,n为当前batch里的笔画数量
            real_input_strokes=torch.zeros((1,256,4)).to(devices)
            real_input_positions=torch.zeros((1,2)).to(devices)
            for batch in range((sketch_stroke_num.size()[0])):
                for stroke_index in range(sketch_stroke_num[batch]):
                    real_input_strokes=torch.cat([real_input_strokes,input_strokes[batch][stroke_index].unsqueeze(0)],dim=0)
                    real_input_positions=torch.cat([real_input_positions,input_positions[batch][stroke_index].unsqueeze(0)])
            real_input_strokes=real_input_strokes[1:]
            real_input_positions=real_input_positions[1:]


            logits,seg_outs,encoder_seg_outs,cluster_seg_outs,graph_seg_outs, hidden_states, attentions = model(real_input_strokes, real_input_positions, real_stroke_point_number, graph_data=graph_data,bool_masked_pos=stroke_mask,sketch_stroke_num=sketch_stroke_num)
            

            #统计笔画的准确率

            _,cur_encoder_seg_corrects=torch.topk(encoder_seg_outs.view(-1,87),k=1,dim=-1)
            
            seg_outs1=cur_encoder_seg_corrects.squeeze(-1).view(-1)
            seg_label1=seg_label.view(-1)
            not_padding_index=torch.where(seg_label1<86)[0]
            final_encoder_seg_outs=torch.index_select(seg_outs1,dim=0,index=not_padding_index)
            final_encoder_seg_label=torch.index_select(seg_label1,dim=0,index=not_padding_index)

            encoder_seg_corrects+=(final_encoder_seg_outs== final_encoder_seg_label).long().sum()
            encoder_seg_num+=final_encoder_seg_label.size()[0]
            
            _, predicts = torch.max(logits, 1)
            
            predicts_accu = torch.sum(predicts == data_batch['category'].to(devices))
            running_corrects += predicts_accu.item()

            _,seg_outs=torch.topk(seg_outs,k=1,dim=2)
            seg_outs=seg_outs.squeeze(2)
            

            final_seg_outs=torch.zeros((1)).to(devices)
            final_seg_label=torch.zeros((1)).to(devices)

            for batch_ in range(sketch_stroke_num.size()[0]):
                cur_sketch_strokes_num=sketch_stroke_num[batch_]
                cur_seg_outs=seg_outs[batch_][:cur_sketch_strokes_num]
                final_seg_outs=torch.cat([final_seg_outs,cur_seg_outs],dim=0)

                final_seg_label=torch.cat([final_seg_label,seg_label[batch_][:cur_sketch_strokes_num]])

            final_seg_outs=final_seg_outs[1:]
            final_seg_label=final_seg_label[1:]
            
            seg_num+=final_seg_label.size()[0]
            
            seg_predicts_accu=torch.sum(final_seg_outs==final_seg_label).item()

            seg_running_corrects +=seg_predicts_accu


            

        # running_corrects = torch.tensor(running_corrects).to(devices)
        seg_running_corrects = seg_running_corrects/(seg_num)
        encoder_seg_acc=encoder_seg_corrects/encoder_seg_num

        # dist.reduce(running_corrects, dst=0)
        # dist.reduce(seg_running_corrects, dst=0)
        return running_corrects,seg_running_corrects,encoder_seg_acc,loss_sum

def train(train_loader, valid_loader, test_loader, model, optim, criterion, devices, opt,seg_criterion,seg_criterion1):
    max_epoch = 200
    best_acc = 0
    iter = 0
    iter_test = 1
    
    # scheduler = StepLR(optim, step_size=3, gamma=0.1)
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=100)
    
    for epoch_id in range(max_epoch):
        print("epoch: "+str(epoch_id+1))
        # train_loader.sampler.set_epoch(epoch_id)
        loss_sum=0
        seg_running_corrects = 0
        seg_num=0

        rec_corrests=0
        rec_num=0

        encoder_seg_corrects=0
        encoder_seg_num=0
        for it, data_batch in enumerate(tqdm(train_loader)):
            
            model.train()
            #b*43*256
            input_strokes = data_batch['points_offset'].to(devices)
            #b*43*2
            input_positions = data_batch['position_list'].to(devices)
            #b*43
            stroke_point_number = data_batch['stroke_number']
            #b*43
            stroke_mask = data_batch['stroke_mask'].to(devices) if opt['mask'] else None
            #b
            sketch_stroke_num=data_batch['sketch_stroke_num'].to(devices)

            graph_data=data_batch['graph_data'].to(devices)

            key_id=data_batch['key_id']

            seg_label_stastic=data_batch['seg_label'].to(devices)

            encoder_seg_label=data_batch['seg_label'].to(devices)
            
            #把每个笔画属于哪个sketch，每个笔画包含多少个point梳理一下，去掉填充，转成nparray
            real_stroke_point_number=[]
            for i in range(stroke_point_number.size()[0]):
                real_stroke_point_number.append(stroke_point_number[i][:sketch_stroke_num[i]].numpy())
            
            
            #从数据集中得到每个笔画的pointoffset和position
            #n*256,n为当前batch里的笔画数量，spg每个草图是256个点
            real_input_strokes=torch.zeros((1,256,4)).to(devices)
            real_input_positions=torch.zeros((1,2)).to(devices)
            for batch in range((sketch_stroke_num.size()[0])):
                for stroke_index in range(sketch_stroke_num[batch]):
                    real_input_strokes=torch.cat([real_input_strokes,input_strokes[batch][stroke_index].unsqueeze(0)],dim=0)
                    real_input_positions=torch.cat([real_input_positions,input_positions[batch][stroke_index].unsqueeze(0)])
            real_input_strokes=real_input_strokes[1:]
            real_input_positions=real_input_positions[1:]

            optim.zero_grad()

            #real_input_strokes,笔画中每个点的偏移
            # real_input_positions,每个笔画第一个点位置
            #real_stroke_point_number，每个笔画的真实点的数量。
            #graph_data，图结构。
            #stroke_mask，None
            #sketch_stroke_num,每个sketch包含多少个笔画
            logits,seg_outs,encoder_seg_outs,cluster_seg_outs,graph_seg_outs, hidden_states, attentions = model(real_input_strokes, real_input_positions, real_stroke_point_number, graph_data=graph_data,bool_masked_pos=stroke_mask,sketch_stroke_num=sketch_stroke_num)
            
            #b*43，43为数据集中单个草图笔画最多的数量，43维度不够，就加了填充
            seg_label_1=data_batch['seg_label']
            
            #下面定义对图神经网络得到的特征直接做分割的标签，是真实的笔画数量
            graph_seg_label=torch.zeros((1))

            #下面是crossentropy的监督矩阵！！！
            #根据当前batch中的最大笔画数量，拿到每个笔画对应的标签
            
            seg_label=torch.full((seg_outs.size()[0],seg_outs.size()[1]),fill_value=86).cuda()
            
            for batch_ in range(sketch_stroke_num.size()[0]):
                
                for stroke_index1 in range(sketch_stroke_num[batch_]):
                    #找到当前笔画的标签
                    stroke_seg_label=seg_label_1[batch_][stroke_index1]
                    # #根据当前壁画的标签找到组件的索引
                    # stroke_seg_label_component_index=torch.nonzero(seg_label_2[batch_]==stroke_seg_label).squeeze()

                    # #设置对应的笔画的组件索引
                    seg_label[batch_][stroke_index1]=stroke_seg_label
                graph_seg_label=torch.cat([graph_seg_label,seg_label_1[batch_][:sketch_stroke_num[batch_]]],dim=0)

            graph_seg_label=graph_seg_label[1:].cuda()
                    
            seg_outs1=seg_outs.view(-1,87)
            
            seg_outs1=torch.log(seg_outs1) 
            
            seg_label=seg_label.view(-1)

            

            #分类
            loss1 = criterion(logits, data_batch['category'].to(devices))
            #mempool分割
            loss2=seg_criterion(seg_outs1,seg_label)

            encoder_seg_outs=encoder_seg_outs.view(-1,87)
            encoder_seg_label=encoder_seg_label.view(-1)
            #下面做encoder_seg的情况
            loss3=seg_criterion1(encoder_seg_outs,encoder_seg_label)

            #下面做key监督的情况
            cluster_label=torch.arange(0,87)
            cluster_label=torch.repeat_interleave(cluster_label,3,dim=0).long().cuda()
            #criterion是crossentropy loss，这里直接用
            loss4=criterion(cluster_seg_outs,cluster_label)

            #下面做graph_seg_outs的损失,这里直接用crossentropy的
            loss5=criterion(graph_seg_outs,graph_seg_label.long())
            
            loss=loss1+10*loss2+20*loss3+loss4+loss5
            #loss_sum用来统计打印的
            loss_sum+=loss.item()
            
            loss.backward()

            optim.step()


            #统计笔画的准确率

            _,cur_encoder_seg_corrects=torch.topk(encoder_seg_outs.view(-1,87),k=1,dim=-1)
            
            seg_outs1=cur_encoder_seg_corrects.squeeze(-1).view(-1)
            seg_label1=seg_label.view(-1)
            not_padding_index=torch.where(seg_label1<86)[0]
            final_encoder_seg_outs=torch.index_select(seg_outs1,dim=0,index=not_padding_index)
            final_encoder_seg_label=torch.index_select(seg_label1,dim=0,index=not_padding_index)

            encoder_seg_corrects+=(final_encoder_seg_outs== final_encoder_seg_label).long().sum()
            encoder_seg_num+=final_encoder_seg_label.size()[0]

            

            #下面统计准确率
            _,seg_outs=torch.topk(seg_outs,k=1,dim=2)
            seg_outs=seg_outs.squeeze(2)
            
            final_seg_outs=torch.zeros((1)).to(devices)
            final_seg_label=torch.zeros((1)).to(devices)

            for batch_ in range(sketch_stroke_num.size()[0]):
                cur_sketch_strokes_num=sketch_stroke_num[batch_]
                cur_seg_outs=seg_outs[batch_][:cur_sketch_strokes_num]
                final_seg_outs=torch.cat([final_seg_outs,cur_seg_outs],dim=0)

                final_seg_label=torch.cat([final_seg_label,seg_label_stastic[batch_][:cur_sketch_strokes_num]])

            final_seg_outs=final_seg_outs[1:]
            final_seg_label=final_seg_label[1:]
            
            seg_num+=final_seg_label.size()[0]
            
            seg_predicts_accu=torch.sum(final_seg_outs==final_seg_label).item()
            seg_running_corrects+=seg_predicts_accu

            _,logits=torch.topk(logits,k=1,dim=1)
            rec_num+=data_batch['category'].size()[0]
            cur_rec_corrests=torch.sum(logits.squeeze()==data_batch['category'].cuda()).item()
            rec_corrests+=cur_rec_corrests


        loss_sum=loss_sum/len(train_loader.dataset)
        
        iter += 1
        
        with open(home + log_folder + train_log_file, 'a') as text_file:
                text_file.write("Time: [%s], Epoch: [%d],loss:%.4f  ,acc: %.4f    %.4f    %.4f\n" % (the_time, epoch_id,loss_sum,rec_corrests/rec_num, seg_running_corrects/seg_num,encoder_seg_corrects/encoder_seg_num))

        if iter % iter_test == 0:
            scheduler.step()
            # dist.barrier()

            acc,seg_acc,encoder_seg_acc,loss_sum1 = test(test_loader, model, devices, opt)
            # acc = test(test_loader, model, devices, opt)
            # if dist.get_rank() == 0:
            acc = acc / len(test_loader.dataset)
            # loss_sum1=loss_sum1/len(test_loader.dataset)
            
            if seg_acc > best_acc:
                best_acc = seg_acc
                save_checkpoint(model, 'best_model.pth')
                best=(the_time, epoch_id,loss_sum1, acc,seg_acc,encoder_seg_acc)
            with open(home + log_folder + log_file, 'a') as text_file:
                text_file.write("Time: [%s], Epoch: [%d],loss:%.4f  ,acc: %.4f   %.4f    %.4f \n" % (the_time, epoch_id,loss_sum1, acc,seg_acc,encoder_seg_acc))
    print(best)
        # if dist.get_rank() == 0:
        # save_checkpoint(model, f'Epoch_{epoch_id}_model')

def main(opt):
    global home
    home = opt['home']
    global log_folder
    log_folder = opt['log_folder']
    
    global ckpt_folder
    ckpt_folder = opt['ckpt_folder']
    
    global max_stroke
    max_stroke = opt['max_stroke']

    batch_size = opt['bs']
    local_rank = opt['local_rank']
    torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl')
    devices = torch.device('cuda', local_rank)
    set_seed(3407)

    # if dist.get_rank() == 0:
    check_all()

    train_dataset = QuickDrawDataset(opt['dataset_path'], 'train')
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=24,shuffle=True)

    valid_dataset = QuickDrawDataset(opt['dataset_path'], 'test')
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=24)

    test_dataset = QuickDrawDataset(opt['dataset_path'], 'test')
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=24)
    
    if opt['pretrain_path'] is None:
        # model = ViTForSketchClassification.from_pretrained('google/vit-base-patch16-224', opt, labels_number=train_dataset.num_categories(), attention_probs_dropout_prob=opt['attention_dropout'], hidden_dropout_prob=opt['embedding_dropout'], use_mask_token=opt['mask']).to(devices)
        print(1)
    else:
        model = ViTForSketchClassification.from_pretrained(opt['pretrain_path'], opt, labels_number=train_dataset.num_categories(), attention_probs_dropout_prob=opt['attention_dropout'], hidden_dropout_prob=opt['embedding_dropout'], use_mask_token=opt['mask']).to(devices)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    # optim = get_optim(model.base_model, opt['lr'], opt['weight_decay'])
    optim =  torch.optim.Adam(model.parameters(),opt['lr'])
    recog_criterion = nn.CrossEntropyLoss().to(devices)
    #用seg_criterion1做分割监督的loss
    #mempool里做过softmax了，这里用log+nllloss。
    seg_criterion = nn.NLLLoss(ignore_index=86).to(devices)
    seg_criterion1 = nn.CrossEntropyLoss(ignore_index=86).to(devices)
    train(train_loader, valid_loader, test_loader, model, optim, recog_criterion, devices, opt,seg_criterion,seg_criterion1)


if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)
