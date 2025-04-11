import dgl
import dgl.data
from dgl.nn import GraphConv, HeteroGraphConv, SAGEConv
from dgl.dataloading import GraphDataLoader
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
from torchmetrics import F1Score, Accuracy, Precision, Recall

from dgl.data import DGLDataset
from torch.utils.data import DataLoader

import os
import numpy as np
import itertools
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)

import logging
from datetime import datetime

# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime('%Y-%m-%d')
log_filename = f'SavedModels/logs/log_{current_time}.log'

# 设置日志配置
logging.basicConfig(
    filename=log_filename,  # 使用时间作为日志文件名
    filemode='a',           # 追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO       # 设置日志级别
)

device = torch.device("cuda")

result_paths = ['Visualization/Address_Label_Pool/', 'Visualization/PU_Learning/','SavedModels/', 'SavedModels/logs']

for rp in result_paths:
    if not os.path.exists(rp+current_time):
        os.makedirs(rp+current_time)
    

def pu_loss(y_pred, y_true, prior=0.1, loss_fn=nn.CrossEntropyLoss(), beta=0.01, gamma=0.01):
    """
    PU learning损失函数实现，使用CrossEntropyLoss。
    """
    positive_mask = (y_true == 1) | (y_true == 0)
    unlabeled_mask = y_true == -1

    # 正类样本的预测与真实标签计算损失
    positive_loss = loss_fn(y_pred[positive_mask], y_true[positive_mask].long())

    # 未标记样本的预测
    unlabeled_preds = y_pred[unlabeled_mask]

    # 正类先验概率调整的未标记损失
    unlabeled_loss_pos = loss_fn(unlabeled_preds, torch.ones(unlabeled_preds.size(0)).long().to(device))
    unlabeled_loss_neg = loss_fn(unlabeled_preds, torch.zeros(unlabeled_preds.size(0)).long().to(device))

    # 总损失 = 正类样本损失 + 未标记样本的正类损失 - 未标记样本的负类损失
    pu_loss = positive_loss + gamma * prior * unlabeled_loss_pos - beta * (1 - prior) * unlabeled_loss_neg 

    return pu_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 预测的概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss


# 双任务图神经网络
class DualGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_node_classes, num_graph_classes, num_edge_features = 6):
        super(DualGNN, self).__init__()
        # 特征提取层
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.batchnorm1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.batchnorm2 = nn.BatchNorm1d(hidden_channels)
        
        # Dropout 层
        # self.dropout = torch.nn.Dropout(0.2)
        
        # 边特征提取层
        self.linear_edge = torch.nn.Linear(num_edge_features, hidden_channels)

        # 分叉
        # 节点分类
        self.convnode1 = GraphConv(hidden_channels, hidden_channels)
        self.convnode2 = GraphConv(hidden_channels, hidden_channels)
        self.node_classifier = torch.nn.Linear(hidden_channels, num_node_classes)

        # 图分类
        self.convgraph1 = GraphConv(hidden_channels+1, hidden_channels)
        # self.convgraph1 = GraphConv(hidden_channels, hidden_channels)
        self.graph_classifier = torch.nn.Linear(hidden_channels, num_graph_classes)

    def forward(self, g, node_feat, nodetype, edge_feat):
        # 将边特征加入图的edata
        g.edata['edge_feat'] = edge_feat.float()

        # 使用边特征来更新消息传递
        def apply_edges(edges):
            # Combine node and edge features appropriately
            return {'m': edges.src['h'] + self.linear_edge(edges.data['edge_feat'])}

        h = self.conv1(g, node_feat)
        h = self.batchnorm1(h)
        h = F.leaky_relu(h)
        
        # h = self.dropout(h)

        h = self.conv2(g, h)
        h = self.batchnorm2(h)
        h = F.leaky_relu(h)

        # # 使用边特征来更新消息传递
        g.ndata['h'] = h
        g.apply_edges(apply_edges) 

        # 节点分类
        hn = self.convnode1(g, h)
        hn = F.leaky_relu(hn)
        hn = self.convnode2(g, hn)
        hn = F.leaky_relu(hn)
        node_out = self.node_classifier(hn)

        # 图分类，加入节点DEX标签特征
        node_feature = nodetype.view(-1, 1)
        hg = torch.cat([h, node_feature], dim=1)
        hg = self.convgraph1(g, hg)
        hg = F.leaky_relu(hg)
        g.ndata['h'] = hg
        hgm = dgl.mean_nodes(g, 'h')
        graph_out = self.graph_classifier(hgm)

        return node_out, graph_out

    # def forward(self, g, node_feat, nodetype, edge_feat):
    #     # g.edata['feature'] = edge_feat.float()
        
    #     h = self.conv1(g, node_feat)
    #     h = self.batchnorm1(h)
        
    #     h = F.leaky_relu(h)
    #     h = self.conv2(g, h)
    #     h = self.batchnorm2(h)
    #     h = F.leaky_relu(h)
        
    #     # 节点分类
    #     hn = self.convnode1(g, h)
    #     # hn = F.leaky_relu(hn)
    #     hn = self.convnode2(g, hn)
    #     node_out = self.node_classifier(hn)
        
    
    #     # 图分类，加入节点DEX标签特征
    #     node_feature = nodetype.view(-1, 1)
    #     hg = torch.cat([h, node_feature], dim=1)
    #     hg = self.convgraph1(g, hg)
    #     # hg = self.convgraph1(g, h)
    #     g.ndata['h'] = hg
    #     hgm = dgl.mean_nodes(g, 'h')
    #     graph_out = self.graph_classifier(hgm)

    #     return node_out, graph_out


def pred(model, average, dataloader, debug=False):
    f1_score = F1Score(num_classes=2, task='binary', average=average)
    precision_score = Precision(num_classes=2, task='binary', average=average)
    recall_score = Recall(num_classes=2, task='binary', average=average)
    accuracy_score = Accuracy(num_classes=2, task='binary', average=average)
    y_pred = torch.tensor([], dtype=torch.float32).to(device)
    y_label = torch.tensor([], dtype=torch.float32).to(device)

    model.eval()
    # 不加这个显存会爆
    with torch.no_grad():
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['feature'].float()
            pred = model(batched_graph, feat)
            # threshold = 0.8  # 调整阈值以控制精确度和召回率的平衡
            # pred = (pred > threshold).float()
            y_pred = torch.cat((y_pred, pred), 0)
            y_label = torch.cat((y_label, labels.to(device)), 0)

    acc = accuracy_score(y_pred.cpu().argmax(1), y_label.cpu())
    f1 = f1_score(y_pred.cpu().argmax(1), y_label.cpu())
    pr = precision_score(y_pred.cpu().argmax(1), y_label.cpu())
    re = recall_score(y_pred.cpu().argmax(1), y_label.cpu())
    # torch.cuda.empty_cache()
    if debug:
        print('Test accuracy:', acc)
        print('Precision: {}, Recall: {}, F1-score: {}'.format(pr, re, f1))
    return y_label, y_pred, acc, pr, re, f1


def train(model, n_epoch, optimizer, scheduler, dataloader, val_dataloader=None, h=None, o=None, nl=None, batch_size=None):
    # criterion = FocalLoss()
    best_f1 = 0
    for epoch in tqdm(range(n_epoch), desc="{} {} {} {}".format(h, o, nl, batch_size)):
        for i, (batched_graph, labels) in enumerate(dataloader):
            batched_graph = batched_graph.to(device)
            model.train()
            feat = batched_graph.ndata['feature'].float()
            optimizer.zero_grad()
            predd = model(batched_graph, feat)
            # loss = F.cross_entropy(predd, labels.to(device), weight=torch.tensor([4.0, 1.0]))
            loss = F.cross_entropy(predd, labels.to(device))
            # loss = criterion(predd, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
        _, _, acc, pr, re, f1 = pred(model, 'weighted', val_dataloader)
        if(f1 > best_f1):
            best_f1 = f1
            # torch.save(model, 'MEV_Classifier.pt'.format(f1))
            # print('Model saved with F1-score: {}'.format(f1))
        if f1 > 0.9:
            print('Val accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}'.format(acc, pr, re, f1))
            torch.save(model, 'SavedModels/{}_{}_{}_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.pt'.format(h, o, nl, batch_size, acc, f1, pr, re))
    # _, _, acc, pr, re, f1 = pred(model, 'weighted', test_dataloader)
    print('Test accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}'.format(acc, pr, re, f1))


def trainDualGNN(model, n_epoch, epoch_thrd, optimizer_list, scheduler, dataloader_list, val_dataloader=None, val_dataloader_node=None, h=None, batch_size=None):

    # 未标记样本历史记录
    # for epoch in tqdm(range(n_epoch), desc="{} {}".format(h, batch_size)):
    for epoch in range(n_epoch):
        # 区分为两个阶段
        # 第一阶段是标签学习，认为unlabelled样本是负样本
        # 第二阶段是PU学习，认为unlabelled样本是无标签样本
        if epoch <= epoch_thrd:
            dataloader = dataloader_list[0]
        else:
            dataloader = dataloader_list[1]
        if epoch <= epoch_thrd:
            optimizer = optimizer_list[0]
        else:
            optimizer = optimizer_list[1]
        # 开始训练
        for i, (batched_graph, graph_labels) in enumerate(dataloader):
            batched_graph = batched_graph.to(device)
            model.train()
            # 节点所有特征
            feat = batched_graph.ndata['feature'].float()
            # 区分为节点特征 和 节点类型
            node_feat = feat[:, :6]
            node_type = feat[:, 6:]
            
            # 边特征
            edge_feat = batched_graph.edata['feature'].float()
            
            # 节点标签
            node_labels = batched_graph.ndata['label'].long()

            optimizer.zero_grad()
            # 正向传播
            node_out, graph_out = model(batched_graph, node_feat, node_type, edge_feat)
            
            if epoch <= epoch_thrd:
                graph_labels[graph_labels == -1] = 0
                graph_loss = F.cross_entropy(graph_out, graph_labels.to(device))
            else:
                graph_loss = pu_loss(graph_out, graph_labels.to(device))
            
            node_loss = F.cross_entropy(node_out, node_labels)
            
            loss = 10 * node_loss + graph_loss
            loss.backward()
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
        if epoch%40 == 0:
            # 两个值分别用于加入PU learning 和 停止训练保存模型
            f1, asw = predDualGNN(model, 'weighted', val_dataloader, val_dataloader_node, epoch)
        
        
        
def predDualGNN(model, average, dataloader, dataloader_node, epoch):
    f1_score = F1Score(num_classes=2, task='binary', average=average)
    precision_score = Precision(num_classes=2, task='binary', average=average)
    recall_score = Recall(num_classes=2, task='binary', average=average)
    accuracy_score = Accuracy(num_classes=2, task='binary', average=average)
    yg_pred = torch.tensor([], dtype=torch.float32).to(device)
    yg_label = torch.tensor([], dtype=torch.float32).to(device)
    yn_pred = torch.tensor([], dtype=torch.float32).to(device)
    yn_label = torch.tensor([], dtype=torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        for batched_graph, graph_labels in dataloader:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['feature'].float()
            # 区分为节点特征 和 节点类型
            node_feat = feat[:, :6]
            node_type = feat[:, 6:]
            
            # 边特征
            edge_feat = batched_graph.edata['feature'].float()
            node_out, graph_out = model(batched_graph, node_feat, node_type, edge_feat)
            
            yg_pred = torch.cat((yg_pred, graph_out), 0)
            yg_label = torch.cat((yg_label, graph_labels.to(device)), 0)
            
            # yn_label = torch.cat((yn_label, batched_graph.ndata['label'].long()), 0)
            
    with torch.no_grad():
        for batched_graph, graph_labels in dataloader_node:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['feature'].float()
            # 区分为节点特征 和 节点类型
            node_feat = feat[:, :6]
            node_type = feat[:, 6:]
            
            # 边特征
            edge_feat = batched_graph.edata['feature'].float()
            node_out, graph_out = model(batched_graph, node_feat, node_type, edge_feat)
            yn_pred = torch.cat((yn_pred, node_out), 0)
            yn_label = torch.cat((yn_label, batched_graph.ndata['label'].long()), 0)

    # 只保留非-1的样本进行准确率验证
    valid_mask = yg_label != -1
    yg_pred_valid = yg_pred[valid_mask]
    yg_label_valid = yg_label[valid_mask]
    # -1 样本中被判别为1的个数
    unlabel_mask = yg_label == -1
    count = torch.sum(yg_pred[unlabel_mask].argmax(1) == 1).item()
    count_unlabel = torch.sum(yg_label == -1).item()
    
    pos_count = torch.sum(yg_label == 1).item()

    acc = accuracy_score(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
    f1 = f1_score(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
    pr = precision_score(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
    re = recall_score(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
    
    accn = accuracy_score(yn_pred.cpu().argmax(1), yn_label.cpu())
    f1n = f1_score(yn_pred.cpu().argmax(1), yn_label.cpu())
    prn = precision_score(yn_pred.cpu().argmax(1), yn_label.cpu())
    ren = recall_score(yn_pred.cpu().argmax(1), yn_label.cpu())
    
    print('Epoch: {:d}, Graph Val accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, Additional: {:d}/{:d}={:.4f}, SW: {:d}'.format(epoch, acc, pr, re, f1,count,count_unlabel, count/count_unlabel, pos_count))
    logging.info('Epoch: {:d}, Graph Val accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, Additional: {:d}/{:d}={:.4f}, SW: {:d}'.format(epoch, acc, pr, re, f1,count,count_unlabel, count/count_unlabel, pos_count))

    print('Epoch: {:d}, Node Val accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}'.format(epoch, accn, prn, ren, f1n))
    logging.info('Epoch: {:d}, Node Val accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}'.format(epoch, accn, prn, ren, f1n))
    
    return f1n, count/len(yg_label)    # return yg_label, yg_pred, acc, pr, re, f1

# 节点分类GNN模型
class NodeGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(NodeGNN, self).__init__()
        # 特征提取层 - 与DualGNN保持一致
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.batchnorm1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.batchnorm2 = nn.BatchNorm1d(hidden_channels)
        
        # 边特征提取层 - 与DualGNN保持一致
        self.linear_edge = torch.nn.Linear(6, hidden_channels)
        
        # 节点分类层 - 与DualGNN保持一致
        self.convnode1 = GraphConv(hidden_channels, hidden_channels)
        self.convnode2 = GraphConv(hidden_channels, hidden_channels)
        self.node_classifier = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, g, node_feat, edge_feat=None):
        # 将边特征加入图的edata
        if edge_feat is not None:
            g.edata['edge_feat'] = edge_feat.float()
        
            # 使用边特征来更新消息传递 - 与DualGNN保持一致
            def apply_edges(edges):
                # Combine node and edge features appropriately
                return {'m': edges.src['h'] + self.linear_edge(edges.data['edge_feat'])}
        
        # 特征提取 - 与DualGNN保持一致
        h = self.conv1(g, node_feat)
        h = self.batchnorm1(h)
        h = F.leaky_relu(h)
        
        h = self.conv2(g, h)
        h = self.batchnorm2(h)
        h = F.leaky_relu(h)
        
        # 边特征处理 - 与DualGNN保持一致
        if edge_feat is not None:
            g.ndata['h'] = h
            g.apply_edges(apply_edges)
        
        # 节点分类 - 与DualGNN保持一致
        hn = self.convnode1(g, h)
        hn = F.leaky_relu(hn)
        hn = self.convnode2(g, hn)
        hn = F.leaky_relu(hn)
        node_out = self.node_classifier(hn)
        
        return node_out

# 图分类GNN模型
class GraphGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_graph_classes, num_edge_features=6):
        super(GraphGNN, self).__init__()
        # 特征提取层
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.batchnorm1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.batchnorm2 = nn.BatchNorm1d(hidden_channels)
        
        # 边特征提取层
        self.linear_edge = torch.nn.Linear(num_edge_features, hidden_channels)
        
        # 图分类层
        self.convgraph1 = GraphConv(hidden_channels+1, hidden_channels)
        self.graph_classifier = torch.nn.Linear(hidden_channels, num_graph_classes)
        
    def forward(self, g, node_feat, edge_feat=None):
        # 将边特征加入图的edata
        if edge_feat is not None:
            g.edata['edge_feat'] = edge_feat.float()
            
            # 使用边特征来更新消息传递
            def apply_edges(edges):
                # Combine node and edge features appropriately
                return {'m': edges.src['h'] + self.linear_edge(edges.data['edge_feat'])}
        
        # 特征提取 - 与DualGNN一致
        h = self.conv1(g, node_feat)
        h = self.batchnorm1(h)
        h = F.leaky_relu(h)
        
        h = self.conv2(g, h)
        h = self.batchnorm2(h)
        h = F.leaky_relu(h)
        
        # 边特征处理 - 与DualGNN一致
        if edge_feat is not None:
            g.ndata['h'] = h
            g.apply_edges(apply_edges)
        
        # 图分类，加入节点DEX标签特征
        # 这里我们假设节点类型信息在特征的最后一维
        node_feature = g.ndata['feature'][:, -1:].float() if 'feature' in g.ndata else torch.zeros((g.num_nodes(), 1), device=h.device)
        hg = torch.cat([h, node_feature], dim=1)
        hg = self.convgraph1(g, hg)
        hg = F.leaky_relu(hg)
        g.ndata['h'] = hg
        hgm = dgl.mean_nodes(g, 'h')
        graph_out = self.graph_classifier(hgm)
        
        return graph_out
