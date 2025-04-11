import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

import torch
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

# device = torch.device("cuda")
device = torch.device("cpu")

from Graph import *
from Data import *

Transfer = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
ERC20Transfer = '0xe59fdd36d0d223c0c7d996db7ad796880f45e1936cb0bb7ac102e7082e031487'
tokenNull = "0x0000000000000000000000000000000000000000"

'''
根据读取后的文件的范围获取待检测列表
'''
def get_event_from_csv(pdtx_df, window_timestamp_start, window_timestamp_end):
    # pdtx_df = pd.read_csv(csv_file)
    pdtx_df_part = pdtx_df[(pdtx_df['blockNumber']>=window_timestamp_start) & (pdtx_df['blockNumber']<=window_timestamp_end)].reset_index(drop=True)
    # 时间段内所有代币转移事件
    window_events={}

    for i in range(len(pdtx_df_part)):
        pdtx = pdtx_df_part.iloc[i,:]
        if len(pdtx['logs']) == 2:
            continue
        logs_json = json.loads(pdtx['logs'].replace("'", '"').replace('False','false'))
        if len(logs_json) > 100:
            continue
        # 判断是否transfer
        for log in logs_json:
            if len(log['topics'])>0:
                if (log['topics'][0]==Transfer) or (log['topics'][0]==ERC20Transfer):
                    # 开始保存
                    cur_log={}
                    txhash = log['transactionHash']
                    if txhash not in window_events:
                        window_events[txhash]={'logIdx':[],'event_list':[]}
                    cur_log["txhash"]=txhash
                    cur_log["blockNumber"]=int(pdtx['blockNumber'])
                    cur_log["topicName"]=log['topics'][0]
                    cur_log["address"]=log['address']
                    cur_log["logIndex"]=log['logIndex']
                    cur_log["data"]=log['data']
                    cur_log["topics"]=log['topics']
                    cur_log["tx_from"]=pdtx['from']
                    cur_log["tx_to"]=pdtx['to']
                    if cur_log["logIndex"] not in window_events[txhash]['logIdx']:
                        window_events[txhash]['event_list'].append(cur_log)
                        window_events[txhash]['logIdx'].append(cur_log["logIndex"])
    return window_events


'''
辅助函数，获取时间戳
'''
def getTimestampByStr(timeStr):
    dt = datetime.fromisoformat(timeStr)
    timestamp = dt.timestamp()
    return timestamp


'''
辅助函数，根据事件进行代币交换解析
'''
def getTokenTransferInfo(event):
    if len(event['topics'])==3:
        token_amount=int(event['data'],16)
        token_from=event['topics'][1]
        token_to=event['topics'][2]
    elif len(event['topics'])==4:
        token_amount=int(event['topics'][3],16)
        token_from=event['topics'][1]
        token_to=event['topics'][2]
    elif len(event['topics'])==1:
        s=event['data'].replace('0x','')
        token_from=s[:64][-40:]
        token_to=s[64:128][-40:]
        token_amount=int(s[-64:],16)
    else:
        return
    
    token_from='0x'+token_from[-40:]
    token_to='0x'+token_to[-40:]
    token_address = event['address']
    return token_address, token_from, token_to, token_amount

'''
检测三明治攻击
'''
def detectSandwichAttackByHeuristic(time_str, pdtx_df):
    all_hash = set([])
    
    # 查重
    def saveCurRes(cur_res):
        res = []
        for i in range(len(cur_res)):
            txhash1 = cur_res[i][0]['txhash']
            txhash2 = cur_res[i][1]['txhash']
            if (hash(txhash1+txhash2) not in all_hash) and (hash(txhash2+txhash1) not in all_hash):
                all_hash.add(hash(txhash1+txhash2))
                all_hash.add(hash(txhash2+txhash1))
                res.append(cur_res[i])
        return res
    
    txt_mode = 'w'
    
    start_blockNumber = int(pdtx_df['blockNumber'].values[0])
    end_blockNumber = int(pdtx_df['blockNumber'].values[-1])
    
    # with open(f'Output/Blockchain/Logs/{time_str}.txt', 'a') as f:
    #     print(start_blockNumber, end_blockNumber, file=f)
        
    tgt_csv = full_path+f"/Heuristic/{time_str}.txt"
    
    for bn in tqdm(range(start_blockNumber, end_blockNumber), desc=time_str):
        cur_res_list = []
        window_events = get_event_from_csv(pdtx_df, bn, bn+2)
        
        # from和to其一相同，内部有相对的代币转移
        for txhash1 in window_events:
            # if txhash1 !='0x799b2b864570872ca921c1da73fbbfc75107171925cb04696b0376f48ecedae4':
            #     continue
            event_list1 = window_events[txhash1]['event_list']
            event_list1_transfer = [getTokenTransferInfo(e1) for e1 in event_list1 if (((e1['topicName'] == Transfer) or (e1['topicName'] == ERC20Transfer)))]
            # 排除各种情况
            event_list1_transfer = [e for e in event_list1_transfer if (e is not None) and (e[1] is not tokenNull)]
            event_list1_transfer = [e for e in event_list1_transfer if e[1] != e[2] ]
            # 只有单笔代币转移则退出
            if len(event_list1_transfer)<=1:
                continue
            tx1_from = event_list1[0]['tx_from']
            tx1_to = event_list1[0]['tx_to']
            for txhash2 in window_events:
                # 寻找from和to相同的两笔交易
                if txhash1 != txhash2:
                    event_list2 = window_events[txhash2]['event_list']
                    event_list2_transfer = [getTokenTransferInfo(e2) for e2 in event_list2 if ((e2['topicName'] == Transfer) or (e2['topicName'] == ERC20Transfer))]
                    event_list2_transfer = [e for e in event_list2_transfer if (e is not None) and (e[1] is not tokenNull)]
                    event_list2_transfer = [e for e in event_list2_transfer if e[1] != e[2]]
                    # 只有单笔代币转移则退出
                    if len(event_list2_transfer)<=1:
                        continue
                    tx2_from = event_list2[0]['tx_from']
                    tx2_to = event_list2[0]['tx_to']

                    if (tx1_from == tx2_from) | (tx1_to == tx2_to):
                        # 判断是否有相对的代币转移
                        # 首先是tx1查找代币转移对
                        addr_list = []
                        for elt1 in event_list1_transfer:
                            addr_list.append(elt1[1])
                            addr_list.append(elt1[2])
                        addr_list = list(set(addr_list))
                        # 查找对子
                        cur_res1 = []
                        for a in addr_list:
                            # 排除必不可能的地址
                            if (a == tx1_to) | (a == tx1_from) | (a == tx2_to) | (a == tx2_from):
                                continue
                            # 找核心地址
                            if (a in [e[1] for e in event_list1_transfer]) and (a in [e[2] for e in event_list1_transfer]):
                                tx1_address1 = [e[0] for e in event_list1_transfer if e[1]==a]# 发送代币类型
                                tx1_address2 = [e[0] for e in event_list1_transfer if e[2]==a]# 接受代币类型
                                # 判断tx2内是否存在
                                if (a in [e[1] for e in event_list2_transfer]) and (a in [e[2] for e in event_list2_transfer]):
                                    tx2_address1 = [e[0] for e in event_list2_transfer if e[1]==a]  # 发送代币类型
                                    tx2_address2 = [e[0] for e in event_list2_transfer if e[2]==a]  # 接受代币类型
                                    # 次序相同则退出
                                    if (tx1_address1 == tx2_address1) and (tx1_address2 == tx2_address2):
                                        continue
                                    
                                    # 是否存在两对代币，而非单一代币
                                    if (set(tx1_address1) & set(tx2_address2)) == (set(tx1_address2) & set(tx2_address1)):
                                        continue
                                    
                                    # 还需要加上次序的判断，而不是仅是否有，不然会有同向的出现
                                    core_1_from = [addr in tx2_address2 for addr in tx1_address1]
                                    core_1_to = [addr in tx2_address1 for addr in tx1_address2]
                                    core_2_from = [addr in tx1_address2 for addr in tx2_address1]
                                    core_2_to = [addr in tx1_address1 for addr in tx2_address2]
                                    if (sum(core_1_from)>0) and (sum(core_1_to)>0) and (sum(core_2_from)>0) and (sum(core_2_to)>0):
                                    
                                        front = {"send_token_type":tx1_address1,
                                                 "send_token_amount":[e[3] for e in event_list1_transfer if e[1]==a],
                                                 "receive_token_type":tx1_address2, 
                                                 "receive_token_amount":[e[3] for e in event_list1_transfer if e[2]==a], 
                                                    "from":[e[1] for e in event_list1_transfer if e[2]==a],
                                                    "to":[e[2] for e in event_list1_transfer if e[2]==a],"txhash":txhash1,
                                                    "coreAddr":a,"blockNumber":event_list1[0]['blockNumber'],
                                                    "tx_from":tx1_from,"tx_to":tx1_to}
                                        back = {"send_token_type":tx2_address1,
                                                "send_token_amount":[e[3] for e in event_list2_transfer if e[1]==a],
                                                "receive_token_type":tx2_address2,
                                                "receive_token_amount":[e[3] for e in event_list2_transfer if e[2]==a],
                                                    "from":[e[1] for e in event_list2_transfer if e[1]==a],
                                                    "to":[e[2] for e in event_list2_transfer if e[1]==a],"txhash":txhash2,
                                                    "coreAddr":a,"blockNumber":event_list2[0]['blockNumber'],
                                                    "tx_from":tx2_from,"tx_to":tx2_to}

                                        
                                        cur_res1.append([front, back])
                        if len(cur_res1)>0:
                            isSaved = saveCurRes(cur_res1)   # 查重
                            if len(isSaved)>0:
                                cur_res_list.append(json.dumps(isSaved))
                                    # break
                        
        with open(tgt_csv, txt_mode) as f:
            for line in cur_res_list:
                f.write(line + '\n')
        txt_mode = 'a'
    return start_blockNumber, end_blockNumber

'''
辅助函数，获取三明治攻击标签集
'''
def getSandwichAttackDF(start_blockNumber, end_blockNumber):
    data_eigenphi_df = pd.read_csv(f'SandwichAttackOnBlockchainTrain/{start_blockNumber//10000*10000}.csv')
    data_eigenphi_df['blockNumber'] = data_eigenphi_df['blockNumber'].apply(lambda x: int(x, 16))
    data_eigenphi_df_part = data_eigenphi_df[(data_eigenphi_df['blockNumber']>start_blockNumber)&(data_eigenphi_df['blockNumber']<end_blockNumber)]
   
    sw_df_part = data_eigenphi_df_part[(data_eigenphi_df_part['sandwichRole']=='FrontRun') |
                      (data_eigenphi_df_part['sandwichRole']=='BackRun') | 
                      (data_eigenphi_df_part['sandwichRole']=='MidFrontRun')] 
    
    
    return sw_df_part


'''
辅助函数，初步验证启发式算法的遗漏
'''
def firstVerify(time_str, pdtx_df, start_blockNumber, end_blockNumber):
    # 分析检测的覆盖率
    sw_df_part = getSandwichAttackDF(start_blockNumber, end_blockNumber)
    
    # 匹配度
    sw_tx_hashes = []

    for i in sw_df_part['txMeta']:
        sw_tx_hashes.append(json.loads(i.replace("'", '"').replace('False','false'))['transactionHash'])
        
    with open(full_path+f"/Heuristic/{time_str}.txt", 'r') as f:
        csw_list = f.readlines()
    
    with open(full_path + f'/Output/Logs/{time_str}.txt', 'a') as f:
        print('Heuristic_Add_Hashes:',len(csw_list), 'total:',len(pdtx_df), file=f)
    csw_tx_hashes = []
    for i in csw_list:
        csw_tx_hashes.append(json.loads(i)[0][0]['txhash'])
        csw_tx_hashes.append(json.loads(i)[0][1]['txhash'])
    
    heuristic_error_list = []
    
    with open(full_path+f'/Output/Error/{time_str}.txt', 'a') as f:
        for i in sw_tx_hashes:
            if i not in csw_tx_hashes:
                heuristic_error_list.append(i)
                print('Heuristic:',i, file=f)
    
    return heuristic_error_list

'''
辅助函数，初始化和获取交易信息
'''
def getTransactionsInfos(time_str):
    # 结果字典预准备
    tx_dict = {}
    with open (full_path+f"/Heuristic/{time_str}.txt", 'r') as f:
        lines = f.readlines()
    for l in lines:
        for i in range(2):
            tx = json.loads(l)[0][i]
            if tx['txhash'] not in tx_dict:
                tx_dict[tx['txhash']] = []
    
    tx_info = getTransactionInfo(tx_dict)
    
    return tx_info, tx_dict

'''
根据交易信息对所有交易构图
'''
def buildTransactionGraphs(tx_info):
    # 构图
    graphs = []
    graphs_hashs = []
    for d in tqdm(range(len(tx_info)),desc='graph'):
        gd = buildGraph(tx_info.iloc[d,:], 'train')
        if gd:
            graphs.append(gd)
            graphs_hashs.append(tx_info.loc[d, 'hash'])
    
    return graphs, graphs_hashs

'''
根据图集构建图数据集，并测试
'''
def buildDatasetAndTest(model, graphs,graphs_hashs, tx_dict):
    # 构建数据集并进行预测
    class TransactionGraphDataset(Dataset):
        def __init__(self, graphs):
            self.graphs = graphs

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            return self.graphs[idx]

    dataset = TransactionGraphDataset(graphs)

    dataloader = GraphDataLoader(dataset, batch_size=1)

    for i, batched_graph in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        feat = batched_graph.ndata['feature'].float()
        node_feat = feat[:, :6]
        node_type = feat[:, 6:]

        edge_feat = batched_graph.edata['feature'].float()

        node_out, graph_out = model(batched_graph, node_feat, node_type, edge_feat)

        argmax_graph = graph_out.argmax(1).cpu().item()  # 单个元素
        argmax_node = node_out.argmax(1).cpu().tolist()   # 多个元素作为列表

        tx_dict[graphs_hashs[i]] = [argmax_graph, argmax_node]
    return tx_dict


'''
模型测试后的第二次验证
'''
def secondVerify(time_str, tx_dict, start_blockNumber, end_blockNumber, heuristic_error_list):
    with open (full_path+f"/Heuristic/{time_str}.txt", 'r') as f:
        lines = f.readlines()
    
    hash_list = []
    for l in lines:
        txs = json.loads(l)[0]
        if len(txs)>1:
            if (len(tx_dict[txs[0]['txhash']]) > 0) and (len(tx_dict[txs[1]['txhash']]) > 0):
                if (tx_dict[txs[0]['txhash']][0] == 1) & (tx_dict[txs[1]['txhash']][0] == 1):
                    hash_list.append(txs[0]['txhash'])
                    hash_list.append(txs[1]['txhash'])
    hash_set = set(hash_list)
    
    sw_df_part = getSandwichAttackDF(start_blockNumber, end_blockNumber)
    
    # 匹配度
    sw_tx_hashes = []
    for i in sw_df_part['txMeta']:
        sw_tx_hashes.append(json.loads(i.replace("'", '"').replace('False','false'))['transactionHash'])

    # 漏检
    with open(full_path+f'/Output/Error/{time_str}.txt', 'a') as f:
        for h in sw_tx_hashes:
            if h not in hash_set:
                if h not in heuristic_error_list:
                    print('GNN:',h,file=f)
            
    # 新检测出来的三明治攻击交易对比已有交易的数量
    add_hash = []

    for h in hash_set:
        if h not in sw_tx_hashes:
            add_hash.append(h)
    # 
    with open(full_path + f'/Output/Logs/{time_str}.txt', 'a') as f:
        print('GNN_Add_Hash:', len(add_hash), 'Label_Hash:', len(sw_tx_hashes), file=f)
    
    hash_list = []
    txinfo_list = []
    for l in lines:
        txs = json.loads(l)[0]
        if (len(tx_dict[txs[0]['txhash']]) > 0) and (len(tx_dict[txs[1]['txhash']]) > 0):
            # if (tx_dict[txs[0]['txhash']][0] == 1) & (tx_dict[txs[1]['txhash']][0] == 1):
            if (tx_dict[txs[0]['txhash']][0] == 1) | (tx_dict[txs[1]['txhash']][0] == 1):
                hash_list.append([txs[0]['txhash'],txs[1]['txhash']])
                txinfo_list.append(txs)
    
    # 新发现的三明治攻击
    res_hash_list = []
    res_txinfo_list = []
    for h, tif in zip(hash_list, txinfo_list):
        if (h[0] not in sw_tx_hashes) or (h[1] not in sw_tx_hashes):
        # if (h[0] not in sw_tx_hashes) and (h[1] not in sw_tx_hashes):
            res_hash_list.append(h)
            res_txinfo_list.append(tif)
    with open(full_path+f'/Output/Logs/{time_str}.txt', 'a') as f:
        print('GNN_Add_Hashes:',len(res_hash_list), file=f)
    
    # with open(full_path+f'/GNN/{time_str}.txt', 'w') as f:
    #     json.dump(res_txinfo_list, f)
    with open(full_path+f'/GNN/{time_str}.txt', 'a') as f:
            for line in res_txinfo_list:
                f.write(json.dumps(line) + '\n')
    
    return res_hash_list, res_txinfo_list

'''
筛选，排除错误
res_hash_list 最后待排除的集合
'''
def lastVerify(pdtx_df, hash_list, txinfo_list, res_dict, tx_info, start_blockNumber, end_blockNumber, detectionType='mempool'):
    # corAddress是DEX地址
    res1_list = []
    res1_list_error = []
    res1_list_error_debug = []
    res1_txinfo = []
    for hashs, txinfo in zip(hash_list, txinfo_list):
        # 核心地址是否交易所地址
        # 保存coreAddr的预测结果
        node_pred_list = []
        node_pred_list_debug = []
        for tx_hash in hashs:
            tx_hash_info = tx_info[tx_info['hash']==tx_hash]
            
            # todo代码量优化
            swtx_logs_json = json.loads(tx_hash_info['logs'].values[0].replace("'", '"').replace('False','false'))
            
            transfer_list = [getTokenTransferInfo(l) for l in swtx_logs_json if ((l['topics'][0] == Transfer) or (l['topics'][0] == ERC20Transfer))]
            # 节点标识
            node_dict = {tx_hash_info['from'].values[0]: 0, tx_hash_info['to'].values[0]: 1}
            node_idx = 2 
            # 针对代币转移事件进行节点编号
            for t in transfer_list:
                if t is None:
                    continue
                if not (t[1] in node_dict):
                    node_dict[t[1]] = node_idx
                    node_idx += 1
                if not (t[2] in node_dict):
                    node_dict[t[2]] = node_idx
                    node_idx += 1
                    
            # 核心地址的序号
            coreAddr_index = node_dict[txinfo[0]['coreAddr']]
            
            # 判断节点是否预测正确
            if coreAddr_index < len(res_dict[tx_hash][1]):
                # if res_dict[tx_hash][1][coreAddr_index] == 1:
                node_pred_list.append(res_dict[tx_hash][1][coreAddr_index])
                node_pred_list_debug.append([res_dict[tx_hash],coreAddr_index])
        # 前后交易都预测准确
        # if True:
        if any(x == 1 for x in node_pred_list):
            res1_list.append(hashs)
            res1_txinfo.append(txinfo)
        else:
            res1_list_error.append(hashs)
            res1_list_error_debug.append(node_pred_list_debug)

    
    with open(full_path+f'/Output/Error/{time_str}.txt', 'a') as f:
        for rle_idx, rle in enumerate(res1_list_error):
            print('Verification_1:', rle, res1_list_error_debug[rle_idx], file=f)
    # 创建代币交换列表，判断两笔对子交易之间是否存在受害者交易(区块链上)
    # if detectionType=='blockchain':
    res2_list = []
    res2_list_error = []
    res2_txinfo = []
    
    window_events = get_event_from_csv(pdtx_df, start_blockNumber, end_blockNumber)
    keys = list(window_events.keys())
    
    for hashs, txinfo in zip(res1_list, res1_txinfo):
        # 核心地址
        coreAddr = txinfo[0]['coreAddr']
        
        # 获取两笔交易中所有代币转移事件
        start_index = keys.index(hashs[0]) + 1
        end_index = keys.index(hashs[1])
        hashs_events = [[getTokenTransferInfo(e) for e in event['event_list']] for event in list(window_events.values())[start_index:end_index]]
        hashs_events_list = []
        # 所有事件
        for he in hashs_events:
            for h in he:
                if h is not None:
                    hashs_events_list.append(h)
        
        # 判断是否有受害交易对存在 
        if (sum([hel[1]==coreAddr for hel in hashs_events_list])>0) and (sum([hel[2]==coreAddr for hel in hashs_events_list])>0):
            if (sum([hel[0] in txinfo[0]['send_token_type'] for hel in hashs_events_list])>0) and (sum([hel[0] in txinfo[1]['send_token_type'] for hel in hashs_events_list])>0):
                res2_list.append(hashs)
                res2_txinfo.append(txinfo)
            else:
                res2_list_error.append(hashs)
        else:
            res2_list_error.append(hashs)
    
    with open(full_path+f'/Output/Error/{time_str}.txt', 'a') as f:
        for r2le in res2_list_error:
            print('Verification_2:', r2le, file=f)
    
    
    # 一笔交易多次出现的话，优先就近原则，其次相近原则
    res3_list = []
    res3_list_dif = []
    res3_list_error = []
    
    res3_txinfo = []
    # 去重
    for res, txinfo in zip(res2_list,res2_txinfo):
        # 两笔交易分别进行判断
        tx1_hash = res[0]
        tx2_hash = res[1]
        tx1_info = tx_info[tx_info['hash']==tx1_hash]
        tx2_info = tx_info[tx_info['hash']==tx2_hash]
        # 区块差
        blocknumber_abs = abs(int(tx2_info['blockNumber'].values[0],16) - int(tx1_info['blockNumber'].values[0],16))
        # 交易序号差
        transactionindex_abs = abs(int(tx2_info['transactionIndex'].values[0],16) - int(tx1_info['transactionIndex'].values[0],16))
        # 两个地址都不存在则保存信息并直接保存
        if not (np.any(np.array(res3_list)==tx1_hash) | (np.any(np.array(res3_list)==tx2_hash))):
            if not ((blocknumber_abs == 0) and transactionindex_abs == 1):
                res3_list.append([res])
                res3_txinfo.append(txinfo)
                res3_list_dif.append([res, blocknumber_abs, transactionindex_abs])
        else:
            # 假如是交易1重复
            if np.any(np.array(res3_list)==tx1_hash):
                rows = np.where(np.array(res3_list) == tx1_hash)[0][0]  # 获取行索引
            else:
                rows = np.where(np.array(res3_list) == tx2_hash)[0][0]  # 获取行索引
            # 对比大小
            # 如果在同一个区块范围内，则都保留
            if (blocknumber_abs == 0) and (res3_list_dif[rows][1]==0):
                res3_list.append([res])
                res3_txinfo.append(txinfo)
                res3_list_dif.append([res, blocknumber_abs, transactionindex_abs])
            elif blocknumber_abs < res3_list_dif[rows][1]:  # 区块更近
                if not ((blocknumber_abs == 0) and transactionindex_abs == 1):
                    res3_list_error.append(res3_list[rows])
                    res3_list[rows] = [res]
                    res3_txinfo[rows] = txinfo
                    res3_list_dif[rows] = [res, blocknumber_abs, transactionindex_abs]
            elif blocknumber_abs == res3_list_dif[rows][1]:  # 区块相同 todo：是否三连三明治
                if transactionindex_abs < res3_list_dif[rows][2]:   # 交易序号更近
                    if not ((blocknumber_abs == 0) and transactionindex_abs == 1):
                        res3_list_error.append(res3_list[rows])
                        res3_list[rows] = [res]
                        res3_txinfo[rows] = txinfo
                        res3_list_dif[rows] = [res, blocknumber_abs, transactionindex_abs]
    with open(full_path+f'/Output/Error/{time_str}.txt', 'a') as f:
        for r3le in res3_list_error:
            print('Verification_3:', r3le, file=f)
            
    
    # 新增验证，跨区块的验证代币汇率
    res4_list = []
    res4_txinfo = []
    res4_list_error = []
    res4_list_dif = []
    
    for res, resd, txinfo, r3ld in zip(res3_list,res3_list_dif,res3_txinfo,res3_list_dif):
        res = res[0]
        if resd[1] == 0:
            res4_list.append(res)
            res4_list_dif.append(r3ld)
            res4_txinfo.append(txinfo)
            continue
        # 判断代币汇率
        tx1_hash = res[0]
        tx2_hash = res[1]
        tx1_info = tx_info[tx_info['hash']==tx1_hash]
        tx2_info = tx_info[tx_info['hash']==tx2_hash]
        WETH = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'
        USDC = '0xdac17f958d2ee523a2206206994597c13d831ec7'
        # 交易顺序
        tx1_index = int(tx1_info['blockNumber'].values[0],16)
        tx2_index = int(tx2_info['blockNumber'].values[0],16)
        # print(txinfo[0])
        tx1_send_amount = float(txinfo[0]['send_token_amount'][0])
        tx1_send_type = txinfo[0]['send_token_type'][0]
        tx1_receive_amount = float(txinfo[0]['receive_token_amount'][0])
        tx1_receive_type = txinfo[0]['receive_token_type'][0]
        tx2_send_amount = float(txinfo[1]['send_token_amount'][0])
        tx2_send_type = txinfo[1]['send_token_type'][0]
        tx2_receive_amount = float(txinfo[1]['receive_token_amount'][0])
        tx2_receive_type = txinfo[1]['receive_token_type'][0]
        
        if (tx1_send_amount != 0) and (tx2_receive_amount != 0):
            if tx1_index < tx2_index:
                # print(1, tx1_receive_amount / tx1_send_amount, tx2_send_amount / tx2_receive_amount)
                if  (tx1_receive_amount / tx1_send_amount) < (tx2_send_amount / tx2_receive_amount):
                    res4_list.append(res)
                    res4_list_dif.append(r3ld)
                    res4_txinfo.append(txinfo)
                else:
                    res4_list_error.append(res)
            else:
                # print(2, tx1_receive_amount / tx1_send_amount, tx2_send_amount / tx2_receive_amount)
                if  (tx1_receive_amount / tx1_send_amount) > (tx2_send_amount / tx2_receive_amount):
                    res4_list.append(res)
                    res4_list_dif.append(r3ld)
                    res4_txinfo.append(txinfo)
                else:
                    res4_list_error.append(res)
        else:
            res4_list_error.append(res)
                
    with open(full_path+f'/Output/Error/{time_str}.txt', 'a') as f:
        for r3le in res4_list_error:
            print('Verification_4:', r3le, file=f)
        
            
    # with open(full_path+f'/Verification/{time_str}.txt', 'w') as f:
    #     json.dump(res3_txinfo, f)
    with open(full_path+f'/Verification/{time_str}.txt', 'a') as f:
            for line in res4_txinfo:
                f.write(json.dumps(line) + '\n')
                
    
    with open(full_path+f'/Output/Logs/{time_str}.txt', 'a') as f:
        print('Verification_1:',len(res1_list), 
          'Verification_2:',len(res2_list), 'Verification_3:',len(res3_list), 'Verification_4:',len(res4_list), file=f)
    
    return res4_list_dif

'''
完整的检测流程：启发式、验证启发式、GNN、验证GNN、最终验证。
'''
def detectSandwich(time_str):
    if not os.path.exists(f'SandwichAttackOnBlockchainTest/{time_str}.csv'):
        return
    pdtx_df = pd.read_csv(f'SandwichAttackOnBlockchainTest/{time_str}.csv')
    if 'blockNumber' not in pdtx_df.columns:
        column_names = ['from', 'to', 'logs', 'txMeta', 'blockNumber', 'transactionIndex', 'gasUsed']
        # 读取没有表头的 CSV 文件并设置自定义列名
        pdtx_df = pd.read_csv(f'SandwichAttackOnBlockchainTest/{time_str}.csv', header=None, names=column_names)
    # pdtx_df = pdtx_df[0:10000]
    pdtx_df['blockNumber'] = pdtx_df['blockNumber'].apply(lambda x: int(x, 16))
    pdtx_df['transactionIndex'] = pdtx_df['transactionIndex'].apply(lambda x: int(x, 16))
    pdtx_df['gasUsed'] = pdtx_df['gasUsed'].apply(lambda x: int(x, 16))
    pdtx_df = pdtx_df.sort_values(by=['blockNumber','transactionIndex']).reset_index(drop=True)
    start_blockNumber = int(pdtx_df['blockNumber'].values[0])
    end_blockNumber = int(pdtx_df['blockNumber'].values[-1])
    
    # if not os.path.exists(f"HeuristicOutput/Blockchain/{time_str}.csv"):
    # 启发式检测
    detectSandwichAttackByHeuristic(time_str, pdtx_df)
    # 验证启发式
    heuristic_error_list = firstVerify(time_str, pdtx_df, start_blockNumber, end_blockNumber)
    
    # 图神经网络部分
    tx_info, tx_dict = getTransactionsInfos(time_str)
    # 完成的构图
    graphs, graphs_hashs = buildTransactionGraphs(tx_info)
    # 读取模型
    model = torch.load('SavedModels/2-0.9016-0.0505.pt',map_location=torch.device('cpu')).to(device)
    # 创建数据集，并测试得到结果
    res_dict = buildDatasetAndTest(model, graphs, graphs_hashs, tx_dict)
    # 验证模型
    res_hash_list, res_txinfo_list = secondVerify(time_str, res_dict, start_blockNumber, end_blockNumber, heuristic_error_list)
    
    # 保存的三项内容为：新增的三明治攻击hash、新增的三明治攻击tx信息（启发式获得的内容）、GNN预测结果
    with open(f'ModelOutput/Blockchain/{time_str}.json', 'w') as f:
        json.dump([res_hash_list, res_txinfo_list, res_dict], f)
    
    with open(f'ModelOutput/Blockchain/{time_str}.json', 'r') as f:
        res_hash_list, res_txinfo_list, res_dict = json.load(f)
    
    # 最后筛选结果
    res_list = lastVerify(pdtx_df, res_hash_list, res_txinfo_list, res_dict, tx_info, start_blockNumber, end_blockNumber)
    
    with open(full_path+f'/Output/Result/{time_str}.txt', 'a') as f:
        for r in res_list:
            print(r, file=f)
            

if __name__ == '__main__':
    # 获取当前日期和时间
    now = datetime.now()

    # 格式化为字符串
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    full_path = f'Output/Blockchain_{current_time_str}'
    # full_path = 'Output/Blockchain_2024-10-16 10:24:07'
    
    if not os.path.exists(full_path):
        os.mkdir(full_path)
        os.mkdir(full_path+'/Heuristic')
        os.mkdir(full_path+'/GNN')
        os.mkdir(full_path+'/Verification')
        os.mkdir(full_path+'/Output')
        os.mkdir(full_path+'/Output/Error')
        os.mkdir(full_path+'/Output/Logs')
        os.mkdir(full_path+'/Output/Result')
    # for time_str in range(19840000, 19910000, 10000):
    # for time_str in range(19300000, 19600000, 10000):
    for time_str in range(19000000, 19300000, 10000):
        detectSandwich(str(time_str))
        # break
    
    
    
    