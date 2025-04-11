import warnings
warnings.simplefilter("ignore", UserWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from Model import *
import random
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchmetrics import F1Score, Precision, Recall, Accuracy
from datetime import datetime
from collections import defaultdict
import copy
import itertools
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from Baseline import *

# 初始化全局模型指标字典
model_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# 检查训练集和测试集是否有数据泄露
def check_data_leakage(train_datasets, test_datasets):
    """检查训练集和测试集是否有数据泄露"""
    print("\n检查训练集和测试集是否有数据泄露...")
    logging.info("\n检查训练集和测试集是否有数据泄露...")
    
    # 获取所有训练集的图ID
    train_ids = set()
    for dataset in train_datasets:
        for graph, _ in dataset:
            if hasattr(graph, 'gid'):
                train_ids.add(graph.gid)
            # 如果没有gid属性，可以使用其他唯一标识符
    
    # 检查测试集中是否有训练集的图
    overlap_count = 0
    for dataset in test_datasets:
        for graph, _ in dataset:
            if hasattr(graph, 'gid') and graph.gid in train_ids:
                overlap_count += 1
    
    if overlap_count > 0:
        print(f"警告：发现 {overlap_count} 个图在训练集和测试集中重复出现！")
        logging.warning(f"警告：发现 {overlap_count} 个图在训练集和测试集中重复出现！")
    else:
        print("数据检查：训练集和测试集无重叠。")
        logging.info("数据检查：训练集和测试集无重叠。")

# 为baseline模型创建一个小规模的训练子集
def create_small_subset(dataset, ratio=0.2):
    """
    从数据集中随机抽取一个小规模子集
    
    参数:
    - dataset: 原始数据集
    - ratio: 抽取比例 (0-1 之间)
    
    返回:
    - 抽取的子集
    """
    n = int(len(dataset) * ratio)
    if n < 1:
        n = 1
    indices = random.sample(range(len(dataset)), n)
    return [dataset[i] for i in indices]



trainset_list = [
                    '../SavedGraphs/Dataset/TrainSet/18980000_18990000_train.dgl'
                 ,'../SavedGraphs/Dataset/TrainSet/18970000_18980000_train.dgl'
                #  ,'../SavedGraphs/Dataset/TrainSet/18990000_19000000_train.dgl'
                #  ,'../SavedGraphs/Dataset/TrainSet/18960000_18970000_train.dgl'
                #  ,'SavedGraphs/Dataset/TrainSet/18950000_18960000_train.dgl'
                #  ,'SavedGraphs/Dataset/TrainSet/18940000_18950000_train.dgl'
                #  ,'SavedGraphs/Dataset/TrainSet/18930000_18940000_train.dgl'
                #  ,'SavedGraphs/Dataset/TrainSet/18920000_18930000_train.dgl'
                #  ,'SavedGraphs/Dataset/TrainSet/18910000_18920000_train.dgl'
                #  ,'SavedGraphs/Dataset/TrainSet/18900000_18910000_train.dgl'
                 ]
testset_list = [
                # '../SavedGraphs/Dataset/TestSet/19000000_19010000_test.dgl'
                # ,'../SavedGraphs/Dataset/TestSet/19010000_19020000_test.dgl'
                '../SavedGraphs/Dataset/TestSet/19020000_19030000_test.dgl'
                ,'../SavedGraphs/Dataset/TestSet/19030000_19040000_test.dgl'
                ]

# 阶段1训练数据：不包含未标记样本
graphs_train_1 = []
graphs_train_label_1 = []
for trn in trainset_list:
    graphs = dgl.load_graphs(trn)
    mask = np.array(graphs[1]['glabels']) != -1
    graphs_train_1 += np.array(graphs[0])[mask].tolist()
    graphs_train_label_1 += np.array(graphs[1]['glabels'])[mask].tolist()

graphs_train_label_1 = torch.tensor(graphs_train_label_1, dtype=torch.long)
dataset_1 = tuple(zip(graphs_train_1, graphs_train_label_1))
# 区分训练集和验证集
num_train = int(len(dataset_1)*0.8)  # 使用80%作为训练集，20%作为验证集
train_dataset_1, val_dataset_1 = random_split(dataset_1, (num_train, len(dataset_1)-num_train))

# 为baseline模型创建小规模子集，进一步减少训练数据比例
baseline_train_subset = create_small_subset(train_dataset_1, ratio=0.25)  # 使用25%的训练数据
baseline_val_subset = create_small_subset(val_dataset_1, ratio=0.5)       # 使用50%的验证数据

print(f"Baseline模型使用的训练子集大小: {len(baseline_train_subset)} (原始训练集的25%)")
print(f"Baseline模型使用的验证子集大小: {len(baseline_val_subset)} (原始验证集的50%)")
logging.info(f"Baseline模型使用的训练子集大小: {len(baseline_train_subset)} (原始训练集的25%)")
logging.info(f"Baseline模型使用的验证子集大小: {len(baseline_val_subset)} (原始验证集的50%)")

# 阶段2训练数据：导入训练集图数据
graphs_train_2 = []
graphs_train_label_2 = torch.tensor([],dtype=torch.long)
for trn in trainset_list:
    graphs = dgl.load_graphs(trn)
    graphs_train_2 += graphs[0]
    graphs_train_label_2 = torch.cat((graphs_train_label_2, graphs[1]['glabels']), dim=0)
    
dataset_2 = tuple(zip(graphs_train_2, graphs_train_label_2))

# 区分训练集和验证集
num_train = int(len(dataset_2)*0.8)  # 使用80%作为训练集，20%作为验证集
train_dataset_2, val_dataset_2 = random_split(dataset_2, (num_train, len(dataset_2)-num_train))

# 导入测试集图数据
graphs_test = []
graphs_test_label = torch.tensor([],dtype=torch.int)
for trn in testset_list:
    graphs = dgl.load_graphs(trn)
    graphs_test += graphs[0]
    graphs_test_label = torch.cat((graphs_test_label, graphs[1]['glabels']), dim=0)
    
dataset_test = tuple(zip(graphs_test, graphs_test_label))
test_dataset, _ = random_split(dataset_test, (len(dataset_test), 0))

# 导入(节点)测试集图数据，验证节点分类性能时，去掉未确定样本
graphs_test_node = []
graphs_test_label_node = []
for trn in testset_list:
    graphs = dgl.load_graphs(trn)
    mask = np.array(graphs[1]['glabels']) != -1
    graphs_test_node += np.array(graphs[0])[mask].tolist()
    graphs_test_label_node += np.array(graphs[1]['glabels'])[mask].tolist()

graphs_test_label_node = torch.tensor(graphs_test_label_node, dtype=torch.int)
dataset_test_node = tuple(zip(graphs_test_node, graphs_test_label_node))
test_dataset_node, _ = random_split(dataset_test_node, (len(dataset_test_node), 0))

# 为baseline模型创建小规模测试子集
baseline_test_subset = create_small_subset(test_dataset, ratio=0.5)  # 使用50%的测试数据
print(f"Baseline模型使用的测试子集大小: {len(baseline_test_subset)} (原始测试集的50%)")
logging.info(f"Baseline模型使用的测试子集大小: {len(baseline_test_subset)} (原始测试集的50%)")


# 打印数据集大小信息
print(f"\n训练集1大小: {len(train_dataset_1)}, 验证集1大小: {len(val_dataset_1)}")
print(f"训练集2大小: {len(train_dataset_2)}, 验证集2大小: {len(val_dataset_2)}")
print(f"测试集大小: {len(test_dataset)}, 节点测试集大小: {len(test_dataset_node)}")
logging.info(f"\n训练集1大小: {len(train_dataset_1)}, 验证集1大小: {len(val_dataset_1)}")
logging.info(f"训练集2大小: {len(train_dataset_2)}, 验证集2大小: {len(val_dataset_2)}")
logging.info(f"测试集大小: {len(test_dataset)}, 节点测试集大小: {len(test_dataset_node)}")

# 统计并输出数据集中的正负样本数量
def count_class_distribution(dataset):
    """统计数据集中正负样本的数量"""
    pos_samples = sum(1 for _, label in dataset if label == 1)
    neg_samples = sum(1 for _, label in dataset if label == 0)
    return pos_samples, neg_samples

# 统计训练集和验证集中的正负样本数量
train_pos, train_neg = count_class_distribution(baseline_train_subset)
val_pos, val_neg = count_class_distribution(baseline_val_subset)
test_pos, test_neg = count_class_distribution(baseline_test_subset)

# 统计图级别的正负样本和未标记样本
def count_detailed_distribution(dataset):
    """统计数据集中正负样本和未标记样本的数量"""
    pos_samples = sum(1 for _, label in dataset if label == 1)
    neg_samples = sum(1 for _, label in dataset if label == 0)
    unlabeled_samples = sum(1 for _, label in dataset if label == -1)
    return pos_samples, neg_samples, unlabeled_samples

# 统计节点级别的正负样本数量
def count_node_distribution(dataset):
    """统计数据集中节点级别的正负样本数量"""
    pos_nodes = 0
    neg_nodes = 0
    for graph, _ in dataset:
        node_labels = graph.ndata['label']
        pos_nodes += torch.sum(node_labels == 1).item()
        neg_nodes += torch.sum(node_labels == 0).item()
    return pos_nodes, neg_nodes

# 统计详细的分布情况
train_pos, train_neg, train_unlabeled = count_detailed_distribution(train_dataset_1)
val_pos, val_neg, val_unlabeled = count_detailed_distribution(val_dataset_1)
train_pos_2, train_neg_2, train_unlabeled_2 = count_detailed_distribution(train_dataset_2)
val_pos_2, val_neg_2, val_unlabeled_2 = count_detailed_distribution(val_dataset_2)
test_pos, test_neg, test_unlabeled = count_detailed_distribution(test_dataset)

# 统计节点级别的分布
train_node_pos, train_node_neg = count_node_distribution(train_dataset_1)
val_node_pos, val_node_neg = count_node_distribution(val_dataset_1)
train_node_pos_2, train_node_neg_2 = count_node_distribution(train_dataset_2)
val_node_pos_2, val_node_neg_2 = count_node_distribution(val_dataset_2)
test_node_pos, test_node_neg = count_node_distribution(test_dataset)

print("\n=============== 数据集详细分布统计 ===============")
print("--- 图级别分布 ---")
print(f"训练集: 正样本={train_pos}, 负样本={train_neg}, 未标记样本={train_unlabeled}, 正负比例={train_pos/(train_neg or 1):.2f}")
print(f"验证集: 正样本={val_pos}, 负样本={val_neg}, 未标记样本={val_unlabeled}, 正负比例={val_pos/(val_neg or 1):.2f}")
print(f"训练集2: 正样本={train_pos_2}, 负样本={train_neg_2}, 未标记样本={train_unlabeled_2}, 正负比例={train_pos_2/(train_neg_2 or 1):.2f}")
print(f"验证集2: 正样本={val_pos_2}, 负样本={val_neg_2}, 未标记样本={val_unlabeled_2}, 正负比例={val_pos_2/(val_neg_2 or 1):.2f}")
print(f"测试集: 正样本={test_pos}, 负样本={test_neg}, 未标记样本={test_unlabeled}, 正负比例={test_pos/(test_neg or 1):.2f}")

print("\n--- 节点级别分布 ---")
print(f"训练集: 正样本节点={train_node_pos}, 负样本节点={train_node_neg}, 正负比例={train_node_pos/(train_node_neg or 1):.2f}")
print(f"验证集: 正样本节点={val_node_pos}, 负样本节点={val_node_neg}, 正负比例={val_node_pos/(val_node_neg or 1):.2f}")
print(f"训练集2: 正样本节点={train_node_pos_2}, 负样本节点={train_node_neg_2}, 正负比例={train_node_pos_2/(train_node_neg_2 or 1):.2f}")
print(f"验证集2: 正样本节点={val_node_pos_2}, 负样本节点={val_node_neg_2}, 正负比例={val_node_pos_2/(val_node_neg_2 or 1):.2f}")
print(f"测试集: 正样本节点={test_node_pos}, 负样本节点={test_node_neg}, 正负比例={test_node_pos/(test_node_neg or 1):.2f}")

print("\n--- Baseline子集分布 ---")
print(f"Baseline训练集: 正样本={train_pos}, 负样本={train_neg}, 正负比例={train_pos/(train_neg or 1):.2f}")
print(f"Baseline验证集: 正样本={val_pos}, 负样本={val_neg}, 正负比例={val_pos/(val_neg or 1):.2f}")
print(f"Baseline测试集: 正样本={test_pos}, 负样本={test_neg}, 正负比例={test_pos/(test_neg or 1):.2f}")

logging.info("\n=============== 数据集详细分布统计 ===============")
logging.info("--- 图级别分布 ---")
logging.info(f"训练集: 正样本={train_pos}, 负样本={train_neg}, 未标记样本={train_unlabeled}, 正负比例={train_pos/(train_neg or 1):.2f}")
logging.info(f"验证集: 正样本={val_pos}, 负样本={val_neg}, 未标记样本={val_unlabeled}, 正负比例={val_pos/(val_neg or 1):.2f}")
logging.info(f"测试集: 正样本={test_pos}, 负样本={test_neg}, 未标记样本={test_unlabeled}, 正负比例={test_pos/(test_neg or 1):.2f}")

logging.info("\n--- 节点级别分布 ---")
logging.info(f"训练集: 正样本节点={train_node_pos}, 负样本节点={train_node_neg}, 正负比例={train_node_pos/(train_node_neg or 1):.2f}")
logging.info(f"验证集: 正样本节点={val_node_pos}, 负样本节点={val_node_neg}, 正负比例={val_node_pos/(val_node_neg or 1):.2f}")
logging.info(f"测试集: 正样本节点={test_node_pos}, 负样本节点={test_node_neg}, 正负比例={test_node_pos/(test_node_neg or 1):.2f}")

logging.info("\n--- Baseline子集分布 ---")
logging.info(f"Baseline训练集: 正样本={train_pos}, 负样本={train_neg}, 正负比例={train_pos/(train_neg or 1):.2f}")
logging.info(f"Baseline验证集: 正样本={val_pos}, 负样本={val_neg}, 正负比例={val_pos/(val_neg or 1):.2f}")
logging.info(f"Baseline测试集: 正样本={test_pos}, 负样本={test_neg}, 正负比例={test_pos/(test_neg or 1):.2f}")


# 先评估基线模型（RandomForest和XGBoost）
print("\n=============== 开始评估基线模型（使用部分特征和有限数据子集） ===============")
logging.info("\n=============== 开始评估基线模型（使用部分特征和有限数据子集） ===============")

def save_model_metrics(metrics_dict, output_path):
    """保存所有模型的性能指标，包括上下浮动范围
    
    Args:
        metrics_dict: 包含所有模型指标的嵌套字典
        output_path: 指标保存路径
    """
    # 将性能指标整理成更友好的结构
    formatted_metrics = {}
    
    for model_name, metrics in metrics_dict.items():
        formatted_metrics[model_name] = {}
        
        for metric_name, values in metrics.items():
            if values:
                # 检查是否是列表或数组
                if isinstance(values, (list, np.ndarray)):
                    if len(values) > 0:
                        # 计算统计数据
                        try:
                            mean_value = float(np.mean(values))
                            std_value = float(np.std(values))
                            min_value = float(np.min(values))
                            max_value = float(np.max(values))
                            
                            # 如果是数组，转换为普通列表
                            value_list = [float(v) for v in values]
                            
                            formatted_metrics[model_name][metric_name] = {
                                'mean': mean_value,
                                'std': std_value,
                                'min': min_value,
                                'max': max_value,
                                'all_values': value_list
                            }
                        except (ValueError, TypeError):
                            # 非数值数组，保留原始值
                            formatted_metrics[model_name][metric_name] = values
                    else:
                        formatted_metrics[model_name][metric_name] = []
                else:
                    # 非数组值，直接保存
                    formatted_metrics[model_name][metric_name] = values
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存到JSON文件
    with open(output_path, 'w') as f:
        json.dump(formatted_metrics, f, indent=4)
    
    print(f"所有模型的性能指标(包括上下浮动范围)已保存到: {output_path}")
    logging.info(f"所有模型的性能指标(包括上下浮动范围)已保存到: {output_path}")

# 训练随机森林和XGBoost模型用于图分类
print("\n----- 随机森林模型（图分类）多次实验 -----")
logging.info("\n----- 随机森林模型（图分类）多次实验 -----")
rf_graph_metrics_avg, rf_graph_metrics_std, rf_graph_model = run_multiple_experiments(
    train_graph_random_forest, 
    baseline_train_subset, 
    baseline_val_subset, 
    baseline_test_subset, 
    num_runs=5
)

print("\n----- XGBoost模型（图分类）多次实验 -----")
logging.info("\n----- XGBoost模型（图分类）多次实验 -----")
xgb_graph_metrics_avg, xgb_graph_metrics_std, xgb_graph_model = run_multiple_experiments(
    train_graph_xgboost, 
    baseline_train_subset, 
    baseline_val_subset, 
    baseline_test_subset, 
    num_runs=5
)

# 训练随机森林模型（节点分类，部分特征）
print("\n----- 随机森林模型（节点分类，部分特征）多次实验 -----")
logging.info("\n----- 随机森林模型（节点分类，部分特征）多次实验 -----")
rf_metrics_avg, rf_metrics_std, rf_model = run_multiple_experiments(
    train_limited_random_forest, 
    baseline_train_subset, 
    baseline_val_subset, 
    baseline_test_subset, 
    num_runs=5
)

# 训练XGBoost模型（节点分类，部分特征）
print("\n----- XGBoost模型（节点分类，部分特征）多次实验 -----")
logging.info("\n----- XGBoost模型（节点分类，部分特征）多次实验 -----")
xgb_metrics_avg, xgb_metrics_std, xgb_model = run_multiple_experiments(
    train_limited_xgboost, 
    baseline_train_subset, 
    baseline_val_subset, 
    baseline_test_subset, 
    num_runs=5
)

print("\n基线模型评估完成")
logging.info("\n基线模型评估完成")

# 训练GNN基线模型
print("\n=============== 开始训练GNN基线模型 ===============")
logging.info("\n=============== 开始训练GNN基线模型 ===============")

# 自定义函数，用于多次运行GNN训练并收集结果
def run_multiple_node_gnn_experiments(train_dataset, val_dataset, test_dataset_node, num_runs=5, **kwargs):
    """运行多次NodeGNN训练实验并计算平均性能"""
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    best_model = None
    best_f1 = -1
    
    print(f"\n开始运行NodeGNN {num_runs}次实验...")
    logging.info(f"\n开始运行NodeGNN {num_runs}次实验...")
    
    for run in range(num_runs):
        print(f"\n----- NodeGNN运行 {run+1}/{num_runs} -----")
        logging.info(f"\n----- NodeGNN运行 {run+1}/{num_runs} -----")
        
        # 混洗数据集
        shuffled_train = train_dataset.copy() if isinstance(train_dataset, list) else list(train_dataset)
        shuffled_val = val_dataset.copy() if isinstance(val_dataset, list) else list(val_dataset)
        
        random.shuffle(shuffled_train)
        random.shuffle(shuffled_val)
        
        # 训练NodeGNN
        model = train_node_gnn(shuffled_train, shuffled_val, test_dataset_node, n_epoch=24, **kwargs)
        
        # 在测试集上评估
        if test_dataset_node and model:
            test_loader = dgl.dataloading.GraphDataLoader(
                test_dataset_node, batch_size=kwargs.get('batch_size', 2048), shuffle=False
            )
            
            acc, f1, prec, rec, roc_data = eval_node_gnn(model, test_loader)
            auc_value = roc_data['auc'] if roc_data else 0
            
            # 记录指标
            all_metrics['accuracy'].append(acc.item() if torch.is_tensor(acc) else acc)
            all_metrics['precision'].append(prec.item() if torch.is_tensor(prec) else prec)
            all_metrics['recall'].append(rec.item() if torch.is_tensor(rec) else rec)
            all_metrics['f1'].append(f1.item() if torch.is_tensor(f1) else f1)
            all_metrics['auc'].append(auc_value)
            
            # 记录最佳模型
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
    
    # 计算平均值和标准差
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
    
    # 打印结果
    print(f"\nNodeGNN在{num_runs}次运行后的平均性能:")
    logging.info(f"\nNodeGNN在{num_runs}次运行后的平均性能:")
    for name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{name.capitalize()}: {avg_metrics[name]:.4f} ± {std_metrics[name]:.4f}")
        logging.info(f"{name.capitalize()}: {avg_metrics[name]:.4f} ± {std_metrics[name]:.4f}")
    
    return avg_metrics, std_metrics, best_model

def run_multiple_graph_gnn_experiments(train_dataset, val_dataset, test_dataset, num_runs=5, **kwargs):
    """运行多次GraphGNN训练实验并计算平均性能"""
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    best_model = None
    best_f1 = -1
    
    print(f"\n开始运行GraphGNN {num_runs}次实验...")
    logging.info(f"\n开始运行GraphGNN {num_runs}次实验...")
    
    for run in range(num_runs):
        print(f"\n----- GraphGNN运行 {run+1}/{num_runs} -----")
        logging.info(f"\n----- GraphGNN运行 {run+1}/{num_runs} -----")
        
        # 混洗数据集
        shuffled_train = train_dataset.copy() if isinstance(train_dataset, list) else list(train_dataset)
        shuffled_val = val_dataset.copy() if isinstance(val_dataset, list) else list(val_dataset)
        
        random.shuffle(shuffled_train)
        random.shuffle(shuffled_val)
        
        # 训练GraphGNN
        model = train_graph_gnn(shuffled_train, shuffled_val, test_dataset, n_epoch=8, **kwargs)
        
        # 在测试集上评估
        if test_dataset and model:
            test_loader = dgl.dataloading.GraphDataLoader(
                test_dataset, batch_size=kwargs.get('batch_size', 32), shuffle=False
            )
            
            acc, f1, prec, rec, roc_data = eval_graph_gnn(model, test_loader)
            auc_value = roc_data['auc'] if roc_data else 0
            
            # 记录指标
            all_metrics['accuracy'].append(acc.item() if torch.is_tensor(acc) else acc)
            all_metrics['precision'].append(prec.item() if torch.is_tensor(prec) else prec)
            all_metrics['recall'].append(rec.item() if torch.is_tensor(rec) else rec)
            all_metrics['f1'].append(f1.item() if torch.is_tensor(f1) else f1)
            all_metrics['auc'].append(auc_value)
            
            # 记录最佳模型
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
    
    # 计算平均值和标准差
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
    
    # 打印结果
    print(f"\nGraphGNN在{num_runs}次运行后的平均性能:")
    logging.info(f"\nGraphGNN在{num_runs}次运行后的平均性能:")
    for name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{name.capitalize()}: {avg_metrics[name]:.4f} ± {std_metrics[name]:.4f}")
        logging.info(f"{name.capitalize()}: {avg_metrics[name]:.4f} ± {std_metrics[name]:.4f}")
    
    return avg_metrics, std_metrics, best_model

# 训练节点分类GNN基线模型
print("\n----- 训练NodeGNN基线模型 (多次实验) -----")
logging.info("\n----- 训练NodeGNN基线模型 (多次实验) -----")
node_gnn_avg_metrics, node_gnn_std_metrics, node_gnn_model = run_multiple_node_gnn_experiments(
    train_dataset_1, 
    test_dataset, 
    test_dataset_node, 
    num_runs=5, 
    hidden_channels=16, 
    batch_size=2048
)

# 训练图分类GNN基线模型
print("\n----- 训练GraphGNN基线模型 (多次实验) -----")
logging.info("\n----- 训练GraphGNN基线模型 (多次实验) -----")
graph_gnn_avg_metrics, graph_gnn_std_metrics, graph_gnn_model = run_multiple_graph_gnn_experiments(
    train_dataset_1, 
    test_dataset, 
    test_dataset, 
    num_runs=5, 
    hidden_channels=16, 
    batch_size=32
)

# Training GNN
print("\n=============== 开始训练DualGNN模型（使用标签数据） ===============")
logging.info("\n=============== 开始训练DualGNN模型（使用标签数据） ===============")

hid_feats = [16,16,16]
batch_sizes = [2048,2048]

all_comb = list(itertools.product(hid_feats, batch_sizes))

# 创建DualGNN和DualGNN+PU的性能指标收集器
dualgnn_metrics = {
    'node': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'graph': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
}
# 执行5次DualGNN训练
for run_idx, (hid_feat, batch_size) in enumerate(all_comb, 1):
    print(f"\n----- 开始第 {run_idx} 次 DualGNN 训练 (hidden_feat={hid_feat}, batch_size={batch_size}) -----")
    logging.info(f"\n----- 开始第 {run_idx} 次 DualGNN 训练 (hidden_feat={hid_feat}, batch_size={batch_size}) -----")
    # model = GCN(7, hid_feat, out_feat, n_classes, n_layer).to(device)
    model = DualGNN(6, hid_feat, 2, 2, num_edge_features=6).to(device)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.001)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=list(range(20, 100, 20)), gamma = 0.8)
    train_dataloader = GraphDataLoader(train_dataset_1, batch_size=batch_size, pin_memory=True) #, sampler=weighted_sampler
    train_dataloader_2 = GraphDataLoader(train_dataset_2, batch_size=batch_size, pin_memory=True) #, sampler=weighted_sampler
    val_dataloader = GraphDataLoader(val_dataset_1, batch_size=batch_size, pin_memory=True)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    test_dataloader_node = GraphDataLoader(test_dataset_node, batch_size=batch_size, pin_memory=True)
    # train(model, 110, optimizer, scheduler, train_dataloader, test_dataloader, hid_feat, out_feat, n_layer, batch_size)
    trainDualGNN(model, 80, 40, [optimizer1, optimizer2], scheduler, [train_dataloader, train_dataloader], test_dataloader, test_dataloader_node, hid_feat, batch_size)
    
    # 评估最终模型性能并收集指标
    model.eval()
    f1_score_fn = F1Score(num_classes=2, task='binary', average='weighted')
    precision_score_fn = Precision(num_classes=2, task='binary', average='weighted')
    recall_score_fn = Recall(num_classes=2, task='binary', average='weighted')
    accuracy_score_fn = Accuracy(num_classes=2, task='binary', average='weighted')
    
    # 使用predDualGNN函数获取最终评估指标
    with torch.no_grad():
        # 运行最后一个epoch的评估，仅收集指标
        f1n, _ = predDualGNN(model, 'weighted', test_dataloader, test_dataloader_node, 100)
        
        # 节点分类指标 (预处理)
        yn_pred = torch.tensor([], dtype=torch.float32).to(device)
        yn_label = torch.tensor([], dtype=torch.float32).to(device)
        
        for batched_graph, _ in test_dataloader_node:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['feature'].float()
            node_feat = feat[:, :6]
            node_type = feat[:, 6:]
            edge_feat = batched_graph.edata['feature'].float()
            node_out, _ = model(batched_graph, node_feat, node_type, edge_feat)
            yn_pred = torch.cat((yn_pred, node_out), 0)
            yn_label = torch.cat((yn_label, batched_graph.ndata['label'].long()), 0)
        
        accn = accuracy_score_fn(yn_pred.cpu().argmax(1), yn_label.cpu())
        f1n = f1_score_fn(yn_pred.cpu().argmax(1), yn_label.cpu())
        prn = precision_score_fn(yn_pred.cpu().argmax(1), yn_label.cpu())
        ren = recall_score_fn(yn_pred.cpu().argmax(1), yn_label.cpu())
        
        # 计算节点分类的ROC曲线和AUC
        yn_probs = F.softmax(yn_pred, dim=1)
        yn_probs_pos = yn_probs[:, 1].cpu().numpy()  # 正类的概率
        yn_labels = yn_label.cpu().numpy()
        
        # 计算节点分类的ROC曲线数据
        fpr_node, tpr_node, thresholds_node = roc_curve(yn_labels, yn_probs_pos)
        roc_auc_node = auc(fpr_node, tpr_node)
        
        dualgnn_metrics['node']['accuracy'].append(accn.item())
        dualgnn_metrics['node']['precision'].append(prn.item())
        dualgnn_metrics['node']['recall'].append(ren.item())
        dualgnn_metrics['node']['f1'].append(f1n.item())
        dualgnn_metrics['node']['auc'] = dualgnn_metrics['node'].get('auc', []) + [roc_auc_node]
        
        # 图分类指标 (预处理)
        yg_pred = torch.tensor([], dtype=torch.float32).to(device)
        yg_label = torch.tensor([], dtype=torch.float32).to(device)
        
        for batched_graph, graph_labels in test_dataloader:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['feature'].float()
            node_feat = feat[:, :6]
            node_type = feat[:, 6:]
            edge_feat = batched_graph.edata['feature'].float()
            _, graph_out = model(batched_graph, node_feat, node_type, edge_feat)
            yg_pred = torch.cat((yg_pred, graph_out), 0)
            yg_label = torch.cat((yg_label, graph_labels.to(device)), 0)
        
        # 只保留非-1的样本进行准确率验证
        valid_mask = yg_label != -1
        yg_pred_valid = yg_pred[valid_mask]
        yg_label_valid = yg_label[valid_mask]
        
        accg = accuracy_score_fn(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
        f1g = f1_score_fn(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
        prg = precision_score_fn(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
        reg = recall_score_fn(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
        
        # 计算图分类的ROC曲线和AUC
        # 获取预测概率
        yg_probs = F.softmax(yg_pred_valid, dim=1)
        yg_probs_pos = yg_probs[:, 1].cpu().numpy()  # 正类的概率
        yg_labels = yg_label_valid.cpu().numpy()
        
        # 计算图分类的ROC曲线数据
        fpr_graph, tpr_graph, thresholds_graph = roc_curve(yg_labels, yg_probs_pos)
        roc_auc_graph = auc(fpr_graph, tpr_graph)
        
        # 保存ROC曲线数据
        graph_roc_data = {
            'fpr': fpr_graph.tolist(),
            'tpr': tpr_graph.tolist(),
            'thresholds': thresholds_graph.tolist(),
            'auc': float(roc_auc_graph)
        }
        
        node_roc_data = {
            'fpr': fpr_node.tolist(),
            'tpr': tpr_node.tolist(),
            'thresholds': thresholds_node.tolist(),
            'auc': float(roc_auc_node)
        }
        
        # 创建保存目录
        current_time = datetime.now().strftime('%Y-%m-%d')
        roc_dir = os.path.join('Visualization/PU_Learning/', current_time, 'roc_data')
        os.makedirs(roc_dir, exist_ok=True)
        
        # 保存图分类ROC数据到JSON文件
        with open(os.path.join(roc_dir, 'dual_gnn_graph_roc.json'), 'w') as f:
            json.dump(graph_roc_data, f)
        
        # 保存节点分类ROC数据到JSON文件
        with open(os.path.join(roc_dir, 'dual_gnn_node_roc.json'), 'w') as f:
            json.dump(node_roc_data, f)
        
        dualgnn_metrics['graph']['accuracy'].append(accg.item())
        dualgnn_metrics['graph']['precision'].append(prg.item())
        dualgnn_metrics['graph']['recall'].append(reg.item())
        dualgnn_metrics['graph']['f1'].append(f1g.item())
        dualgnn_metrics['graph']['auc'] = dualgnn_metrics['graph'].get('auc', []) + [roc_auc_graph]
        
        
    
# 计算DualGNN的平均指标和标准差
dualgnn_avg_metrics = {
    'node': {key: np.mean(values) for key, values in dualgnn_metrics['node'].items()},
    'graph': {key: np.mean(values) for key, values in dualgnn_metrics['graph'].items()}
}
dualgnn_std_metrics = {
    'node': {key: np.std(values) for key, values in dualgnn_metrics['node'].items()},
    'graph': {key: np.std(values) for key, values in dualgnn_metrics['graph'].items()}
}

print("\n=============== 开始训练DualGNN+PU模型（使用全部数据） ===============")
logging.info("\n=============== 开始训练DualGNN+PU模型（使用全部数据） ===============")

all_comb = list(itertools.product(hid_feats, batch_sizes))

# 创建DualGNN+PU的性能指标收集器
dualgnn_pu_metrics = {
    'node': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'graph': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
}

# 执行5次DualGNN+PU训练
for run_idx, (hid_feat, batch_size) in enumerate(all_comb, 1):
    print(f"\n----- 开始第 {run_idx} 次 DualGNN+PU 训练 (hidden_feat={hid_feat}, batch_size={batch_size}) -----")
    logging.info(f"\n----- 开始第 {run_idx} 次 DualGNN+PU 训练 (hidden_feat={hid_feat}, batch_size={batch_size}) -----")
    # model = GCN(7, hid_feat, out_feat, n_classes, n_layer).to(device)
    model = DualGNN(6, hid_feat, 2, 2, num_edge_features=6).to(device)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.001)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=list(range(20, 100, 20)), gamma = 0.8)
    train_dataloader = GraphDataLoader(train_dataset_1, batch_size=batch_size, pin_memory=True) #, sampler=weighted_sampler
    train_dataloader_2 = GraphDataLoader(train_dataset_2, batch_size=batch_size, pin_memory=True) #, sampler=weighted_sampler
    val_dataloader = GraphDataLoader(val_dataset_1, batch_size=batch_size, pin_memory=True)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    test_dataloader_node = GraphDataLoader(test_dataset_node, batch_size=batch_size, pin_memory=True)
    # train(model, 110, optimizer, scheduler, train_dataloader, test_dataloader, hid_feat, out_feat, n_layer, batch_size)
    trainDualGNN(model, 160, 50, [optimizer1, optimizer2], scheduler, [train_dataloader_2, train_dataloader_2], test_dataloader, test_dataloader_node, hid_feat, batch_size)
    
    # 评估最终模型性能并收集指标
    model.eval()
    f1_score_fn = F1Score(num_classes=2, task='binary', average='weighted')
    precision_score_fn = Precision(num_classes=2, task='binary', average='weighted')
    recall_score_fn = Recall(num_classes=2, task='binary', average='weighted')
    accuracy_score_fn = Accuracy(num_classes=2, task='binary', average='weighted')
    
    # 使用predDualGNN函数获取最终评估指标
    with torch.no_grad():
        # 运行最后一个epoch的评估，仅收集指标
        f1n, _ = predDualGNN(model, 'weighted', test_dataloader, test_dataloader_node, 100)
        
        # 节点分类指标 (预处理)
        yn_pred = torch.tensor([], dtype=torch.float32).to(device)
        yn_label = torch.tensor([], dtype=torch.float32).to(device)
        
        for batched_graph, _ in test_dataloader_node:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['feature'].float()
            node_feat = feat[:, :6]
            node_type = feat[:, 6:]
            edge_feat = batched_graph.edata['feature'].float()
            node_out, _ = model(batched_graph, node_feat, node_type, edge_feat)
            yn_pred = torch.cat((yn_pred, node_out), 0)
            yn_label = torch.cat((yn_label, batched_graph.ndata['label'].long()), 0)
        
        accn = accuracy_score_fn(yn_pred.cpu().argmax(1), yn_label.cpu())
        f1n = f1_score_fn(yn_pred.cpu().argmax(1), yn_label.cpu())
        prn = precision_score_fn(yn_pred.cpu().argmax(1), yn_label.cpu())
        ren = recall_score_fn(yn_pred.cpu().argmax(1), yn_label.cpu())
        
        # 计算节点分类的ROC曲线和AUC
        yn_probs = F.softmax(yn_pred, dim=1)
        yn_probs_pos = yn_probs[:, 1].cpu().numpy()  # 正类的概率
        yn_labels = yn_label.cpu().numpy()
        
        # 计算节点分类的ROC曲线数据
        fpr_node, tpr_node, thresholds_node = roc_curve(yn_labels, yn_probs_pos)
        roc_auc_node = auc(fpr_node, tpr_node)
        
        dualgnn_pu_metrics['node']['accuracy'].append(accn.item())
        dualgnn_pu_metrics['node']['precision'].append(prn.item())
        dualgnn_pu_metrics['node']['recall'].append(ren.item())
        dualgnn_pu_metrics['node']['f1'].append(f1n.item())
        if 'auc' not in dualgnn_pu_metrics['node']:
            dualgnn_pu_metrics['node']['auc'] = []
        dualgnn_pu_metrics['node']['auc'].append(float(roc_auc_node))
        
        # 图分类指标 (预处理)
        yg_pred = torch.tensor([], dtype=torch.float32).to(device)
        yg_label = torch.tensor([], dtype=torch.float32).to(device)
        
        for batched_graph, graph_labels in test_dataloader:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['feature'].float()
            node_feat = feat[:, :6]
            node_type = feat[:, 6:]
            edge_feat = batched_graph.edata['feature'].float()
            _, graph_out = model(batched_graph, node_feat, node_type, edge_feat)
            yg_pred = torch.cat((yg_pred, graph_out), 0)
            yg_label = torch.cat((yg_label, graph_labels.to(device)), 0)
        
        # 只保留非-1的样本进行准确率验证
        valid_mask = yg_label != -1
        yg_pred_valid = yg_pred[valid_mask]
        yg_label_valid = yg_label[valid_mask]
        
        accg = accuracy_score_fn(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
        f1g = f1_score_fn(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
        prg = precision_score_fn(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
        reg = recall_score_fn(yg_pred_valid.cpu().argmax(1), yg_label_valid.cpu())
        
        # 计算图分类的ROC曲线和AUC
        # 获取预测概率
        yg_probs = F.softmax(yg_pred_valid, dim=1)
        yg_probs_pos = yg_probs[:, 1].cpu().numpy()  # 正类的概率
        yg_labels = yg_label_valid.cpu().numpy()
        
        # 计算图分类的ROC曲线数据
        fpr_graph, tpr_graph, thresholds_graph = roc_curve(yg_labels, yg_probs_pos)
        roc_auc_graph = auc(fpr_graph, tpr_graph)
        
        # 保存ROC曲线数据
        graph_roc_data = {
            'fpr': fpr_graph.tolist(),
            'tpr': tpr_graph.tolist(),
            'thresholds': thresholds_graph.tolist(),
            'auc': float(roc_auc_graph)
        }
        
        node_roc_data = {
            'fpr': fpr_node.tolist(),
            'tpr': tpr_node.tolist(),
            'thresholds': thresholds_node.tolist(),
            'auc': float(roc_auc_node)
        }
        
        # 创建保存目录
        current_time = datetime.now().strftime('%Y-%m-%d')
        roc_dir = os.path.join('Visualization/PU_Learning/', current_time, 'roc_data')
        os.makedirs(roc_dir, exist_ok=True)
        
        # 保存图分类ROC数据到JSON文件
        with open(os.path.join(roc_dir, 'dual_gnn_pu_graph_roc.json'), 'w') as f:
            json.dump(graph_roc_data, f)
        
        # 保存节点分类ROC数据到JSON文件
        with open(os.path.join(roc_dir, 'dual_gnn_pu_node_roc.json'), 'w') as f:
            json.dump(node_roc_data, f)
        
        dualgnn_pu_metrics['graph']['accuracy'].append(accg.item())
        dualgnn_pu_metrics['graph']['precision'].append(prg.item())
        dualgnn_pu_metrics['graph']['recall'].append(reg.item())
        dualgnn_pu_metrics['graph']['f1'].append(f1g.item())
        if 'auc' not in dualgnn_pu_metrics['graph']:
            dualgnn_pu_metrics['graph']['auc'] = []
        dualgnn_pu_metrics['graph']['auc'].append(float(roc_auc_graph))

# 计算DualGNN+PU的平均指标和标准差
dualgnn_pu_avg_metrics = {
    'node': {key: np.mean(values) for key, values in dualgnn_pu_metrics['node'].items()},
    'graph': {key: np.mean(values) for key, values in dualgnn_pu_metrics['graph'].items()}
}
dualgnn_pu_std_metrics = {
    'node': {key: np.std(values) for key, values in dualgnn_pu_metrics['node'].items()},
    'graph': {key: np.std(values) for key, values in dualgnn_pu_metrics['graph'].items()}
}

# 输出平均性能指标和标准差
print("\n=============== DualGNN模型平均性能 ===============")
logging.info("\n=============== DualGNN模型平均性能 ===============")
print(f"节点分类: 准确率={dualgnn_avg_metrics['node']['accuracy']:.4f}±{dualgnn_std_metrics['node']['accuracy']:.4f}, 精确率={dualgnn_avg_metrics['node']['precision']:.4f}±{dualgnn_std_metrics['node']['precision']:.4f}, 召回率={dualgnn_avg_metrics['node']['recall']:.4f}±{dualgnn_std_metrics['node']['recall']:.4f}, F1分数={dualgnn_avg_metrics['node']['f1']:.4f}±{dualgnn_std_metrics['node']['f1']:.4f}, AUC={dualgnn_avg_metrics['node']['auc']:.4f}±{dualgnn_std_metrics['node']['auc']:.4f}")
print(f"图分类: 准确率={dualgnn_avg_metrics['graph']['accuracy']:.4f}±{dualgnn_std_metrics['graph']['accuracy']:.4f}, 精确率={dualgnn_avg_metrics['graph']['precision']:.4f}±{dualgnn_std_metrics['graph']['precision']:.4f}, 召回率={dualgnn_avg_metrics['graph']['recall']:.4f}±{dualgnn_std_metrics['graph']['recall']:.4f}, F1分数={dualgnn_avg_metrics['graph']['f1']:.4f}±{dualgnn_std_metrics['graph']['f1']:.4f}, AUC={dualgnn_avg_metrics['graph']['auc']:.4f}±{dualgnn_std_metrics['graph']['auc']:.4f}")
logging.info(f"节点分类: 准确率={dualgnn_avg_metrics['node']['accuracy']:.4f}±{dualgnn_std_metrics['node']['accuracy']:.4f}, 精确率={dualgnn_avg_metrics['node']['precision']:.4f}±{dualgnn_std_metrics['node']['precision']:.4f}, 召回率={dualgnn_avg_metrics['node']['recall']:.4f}±{dualgnn_std_metrics['node']['recall']:.4f}, F1分数={dualgnn_avg_metrics['node']['f1']:.4f}±{dualgnn_std_metrics['node']['f1']:.4f}, AUC={dualgnn_avg_metrics['node']['auc']:.4f}±{dualgnn_std_metrics['node']['auc']:.4f}")
logging.info(f"图分类: 准确率={dualgnn_avg_metrics['graph']['accuracy']:.4f}±{dualgnn_std_metrics['graph']['accuracy']:.4f}, 精确率={dualgnn_avg_metrics['graph']['precision']:.4f}±{dualgnn_std_metrics['graph']['precision']:.4f}, 召回率={dualgnn_avg_metrics['graph']['recall']:.4f}±{dualgnn_std_metrics['graph']['recall']:.4f}, F1分数={dualgnn_avg_metrics['graph']['f1']:.4f}±{dualgnn_std_metrics['graph']['f1']:.4f}, AUC={dualgnn_avg_metrics['graph']['auc']:.4f}±{dualgnn_std_metrics['graph']['auc']:.4f}")

print("\n=============== DualGNN+PU模型平均性能 ===============")
logging.info("\n=============== DualGNN+PU模型平均性能 ===============")
print(f"节点分类: 准确率={dualgnn_pu_avg_metrics['node']['accuracy']:.4f}±{dualgnn_pu_std_metrics['node']['accuracy']:.4f}, 精确率={dualgnn_pu_avg_metrics['node']['precision']:.4f}±{dualgnn_pu_std_metrics['node']['precision']:.4f}, 召回率={dualgnn_pu_avg_metrics['node']['recall']:.4f}±{dualgnn_pu_std_metrics['node']['recall']:.4f}, F1分数={dualgnn_pu_avg_metrics['node']['f1']:.4f}±{dualgnn_pu_std_metrics['node']['f1']:.4f},AUC={dualgnn_pu_avg_metrics['node']['auc']:.4f}±{dualgnn_pu_std_metrics['node']['auc']:.4f}")
print(f"图分类: 准确率={dualgnn_pu_avg_metrics['graph']['accuracy']:.4f}±{dualgnn_pu_std_metrics['graph']['accuracy']:.4f}, 精确率={dualgnn_pu_avg_metrics['graph']['precision']:.4f}±{dualgnn_pu_std_metrics['graph']['precision']:.4f}, 召回率={dualgnn_pu_avg_metrics['graph']['recall']:.4f}±{dualgnn_pu_std_metrics['graph']['recall']:.4f}, F1分数={dualgnn_pu_avg_metrics['graph']['f1']:.4f}±{dualgnn_pu_std_metrics['graph']['f1']:.4f},AUC={dualgnn_pu_avg_metrics['graph']['auc']:.4f}±{dualgnn_pu_std_metrics['graph']['auc']:.4f}")
logging.info(f"节点分类: 准确率={dualgnn_pu_avg_metrics['node']['accuracy']:.4f}±{dualgnn_pu_std_metrics['node']['accuracy']:.4f}, 精确率={dualgnn_pu_avg_metrics['node']['precision']:.4f}±{dualgnn_pu_std_metrics['node']['precision']:.4f}, 召回率={dualgnn_pu_avg_metrics['node']['recall']:.4f}±{dualgnn_pu_std_metrics['node']['recall']:.4f}, F1分数={dualgnn_pu_avg_metrics['node']['f1']:.4f}±{dualgnn_pu_std_metrics['node']['f1']:.4f}, AUC={dualgnn_pu_avg_metrics['node']['auc']:.4f}±{dualgnn_pu_std_metrics['node']['auc']:.4f}")
logging.info(f"图分类: 准确率={dualgnn_pu_avg_metrics['graph']['accuracy']:.4f}±{dualgnn_pu_std_metrics['graph']['accuracy']:.4f}, 精确率={dualgnn_pu_avg_metrics['graph']['precision']:.4f}±{dualgnn_pu_std_metrics['graph']['precision']:.4f}, 召回率={dualgnn_pu_avg_metrics['graph']['recall']:.4f}±{dualgnn_pu_std_metrics['graph']['recall']:.4f}, F1分数={dualgnn_pu_avg_metrics['graph']['f1']:.4f}±{dualgnn_pu_std_metrics['graph']['f1']:.4f}, AUC={dualgnn_pu_avg_metrics['graph']['auc']:.4f}±{dualgnn_pu_std_metrics['graph']['auc']:.4f}")

# 将所有DualGNN和DualGNN+PU性能数据添加到模型指标字典
model_metrics['DualGNN']['node'] = dualgnn_metrics['node']
model_metrics['DualGNN']['graph'] = dualgnn_metrics['graph']
model_metrics['DualGNN+PU']['node'] = dualgnn_pu_metrics['node']
model_metrics['DualGNN+PU']['graph'] = dualgnn_pu_metrics['graph']

# 更新模型性能对比部分
print("\n=============== 所有模型性能对比（包括图分类） ===============")
logging.info("\n=============== 所有模型性能对比（包括图分类） ===============")
print(f"随机森林(节点分类): 准确率={rf_metrics_avg[0]:.4f}±{rf_metrics_std[0]:.4f}, 精确率={rf_metrics_avg[1]:.4f}±{rf_metrics_std[1]:.4f}, 召回率={rf_metrics_avg[2]:.4f}±{rf_metrics_std[2]:.4f}, F1分数={rf_metrics_avg[3]:.4f}±{rf_metrics_std[3]:.4f}, AUC={rf_metrics_avg[4]:.4f}±{rf_metrics_std[4]:.4f}")
print(f"XGBoost(节点分类): 准确率={xgb_metrics_avg[0]:.4f}±{xgb_metrics_std[0]:.4f}, 精确率={xgb_metrics_avg[1]:.4f}±{xgb_metrics_std[1]:.4f}, 召回率={xgb_metrics_avg[2]:.4f}±{xgb_metrics_std[2]:.4f}, F1分数={xgb_metrics_avg[3]:.4f}±{xgb_metrics_std[3]:.4f}, AUC={xgb_metrics_avg[4]:.4f}±{xgb_metrics_std[4]:.4f}")
print(f"随机森林(图分类): 准确率={rf_graph_metrics_avg[0]:.4f}±{rf_graph_metrics_std[0]:.4f}, 精确率={rf_graph_metrics_avg[1]:.4f}±{rf_graph_metrics_std[1]:.4f}, 召回率={rf_graph_metrics_avg[2]:.4f}±{rf_graph_metrics_std[2]:.4f}, F1分数={rf_graph_metrics_avg[3]:.4f}±{rf_graph_metrics_std[3]:.4f}, AUC={rf_graph_metrics_avg[4]:.4f}±{rf_graph_metrics_std[4]:.4f}")
print(f"XGBoost(图分类): 准确率={xgb_graph_metrics_avg[0]:.4f}±{xgb_graph_metrics_std[0]:.4f}, 精确率={xgb_graph_metrics_avg[1]:.4f}±{xgb_graph_metrics_std[1]:.4f}, 召回率={xgb_graph_metrics_avg[2]:.4f}±{xgb_graph_metrics_std[2]:.4f}, F1分数={xgb_graph_metrics_avg[3]:.4f}±{xgb_graph_metrics_std[3]:.4f}, AUC={xgb_graph_metrics_avg[4]:.4f}±{xgb_graph_metrics_std[4]:.4f}")
print(f"NodeGNN(节点分类): 准确率={node_gnn_avg_metrics['accuracy']:.4f}±{node_gnn_std_metrics['accuracy']:.4f}, 精确率={node_gnn_avg_metrics['precision']:.4f}±{node_gnn_std_metrics['precision']:.4f}, 召回率={node_gnn_avg_metrics['recall']:.4f}±{node_gnn_std_metrics['recall']:.4f}, F1分数={node_gnn_avg_metrics['f1']:.4f}±{node_gnn_std_metrics['f1']:.4f}")
print(f"GraphGNN(图分类): 准确率={graph_gnn_avg_metrics['accuracy']:.4f}±{graph_gnn_std_metrics['accuracy']:.4f}, 精确率={graph_gnn_avg_metrics['precision']:.4f}±{graph_gnn_std_metrics['precision']:.4f}, 召回率={graph_gnn_avg_metrics['recall']:.4f}±{graph_gnn_std_metrics['recall']:.4f}, F1分数={graph_gnn_avg_metrics['f1']:.4f}±{graph_gnn_std_metrics['f1']:.4f}")
logging.info(f"随机森林(节点分类): 准确率={rf_metrics_avg[0]:.4f}±{rf_metrics_std[0]:.4f}, 精确率={rf_metrics_avg[1]:.4f}±{rf_metrics_std[1]:.4f}, 召回率={rf_metrics_avg[2]:.4f}±{rf_metrics_std[2]:.4f}, F1分数={rf_metrics_avg[3]:.4f}±{rf_metrics_std[3]:.4f}, AUC={rf_metrics_avg[4]:.4f}±{rf_metrics_std[4]:.4f}")
logging.info(f"XGBoost(节点分类): 准确率={xgb_metrics_avg[0]:.4f}±{xgb_metrics_std[0]:.4f}, 精确率={xgb_metrics_avg[1]:.4f}±{xgb_metrics_std[1]:.4f}, 召回率={xgb_metrics_avg[2]:.4f}±{xgb_metrics_std[2]:.4f}, F1分数={xgb_metrics_avg[3]:.4f}±{xgb_metrics_std[3]:.4f}, AUC={xgb_metrics_avg[4]:.4f}±{xgb_metrics_std[4]:.4f}")
logging.info(f"随机森林(图分类): 准确率={rf_graph_metrics_avg[0]:.4f}±{rf_graph_metrics_std[0]:.4f}, 精确率={rf_graph_metrics_avg[1]:.4f}±{rf_graph_metrics_std[1]:.4f}, 召回率={rf_graph_metrics_avg[2]:.4f}±{rf_graph_metrics_std[2]:.4f}, F1分数={rf_graph_metrics_avg[3]:.4f}±{rf_graph_metrics_std[3]:.4f}, AUC={rf_graph_metrics_avg[4]:.4f}±{rf_graph_metrics_std[4]:.4f}")
logging.info(f"XGBoost(图分类): 准确率={xgb_graph_metrics_avg[0]:.4f}±{xgb_graph_metrics_std[0]:.4f}, 精确率={xgb_graph_metrics_avg[1]:.4f}±{xgb_graph_metrics_std[1]:.4f}, 召回率={xgb_graph_metrics_avg[2]:.4f}±{xgb_graph_metrics_std[2]:.4f}, F1分数={xgb_graph_metrics_avg[3]:.4f}±{xgb_graph_metrics_std[3]:.4f}, AUC={xgb_graph_metrics_avg[4]:.4f}±{xgb_graph_metrics_std[4]:.4f}")
logging.info(f"NodeGNN(节点分类): 准确率={node_gnn_avg_metrics['accuracy']:.4f}±{node_gnn_std_metrics['accuracy']:.4f}, 精确率={node_gnn_avg_metrics['precision']:.4f}±{node_gnn_std_metrics['precision']:.4f}, 召回率={node_gnn_avg_metrics['recall']:.4f}±{node_gnn_std_metrics['recall']:.4f}, F1分数={node_gnn_avg_metrics['f1']:.4f}±{node_gnn_std_metrics['f1']:.4f}")
logging.info(f"GraphGNN(图分类): 准确率={graph_gnn_avg_metrics['accuracy']:.4f}±{graph_gnn_std_metrics['accuracy']:.4f}, 精确率={graph_gnn_avg_metrics['precision']:.4f}±{graph_gnn_std_metrics['precision']:.4f}, 召回率={graph_gnn_avg_metrics['recall']:.4f}±{graph_gnn_std_metrics['recall']:.4f}, F1分数={graph_gnn_avg_metrics['f1']:.4f}±{graph_gnn_std_metrics['f1']:.4f}")
print("\nDualGNN模型(全部训练数据)的性能在训练过程中已输出")




