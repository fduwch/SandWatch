import requests
import json
import pandas as pd
import glob
import torch as th
import time
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

url = "http://10.176.35.56:8545"
headers = {
  'Content-Type': 'application/json'
}


# 判断一个地址是不是合约地址，用于一部分特征
def isContract(address):
  payload = json.dumps({
    "method": "eth_getCode",
    "params": [
      address,
      "latest"
    ],
    "id": 1,
    "jsonrpc": "2.0"
  })
  response = requests.request("POST", url, headers=headers, data=payload)
  if response.json()['result'] is not None:
    if response.json()['result'] != '0x':
      return True
    else:
      return False
# isContract('0x388C818CA8B9251b393131C08a736A67ccB19297')

# 判断一个地址是不是被标记为公开的Dexs
with open('labeled_dataset/dexs.json', 'r') as f:
  dexs_list = json.load(f)
  dexs_list = [d.lower() for d in dexs_list]
def isDexs(address):
  return address.lower() in dexs_list
# isDexs('0xe66B31678d6C16E9ebf358268A790B763C133750')

# 判断一个地址是不是MEV-bots
with open('labeled_dataset/mev_bots.json', 'r') as f:
  mev_bots_list = json.load(f)
  mev_bots_list = [m.lower() for m in mev_bots_list]
def isMEVBots(address):
  return address.lower() in mev_bots_list
# isMEVBots('0x00000007f7a9056880D057f611e80c419f9b20c8')

# 判断一个地址是不是SandwichAttacker, CommonTrader, SandwichMiddleAttacker
with open('labeled_dataset/tags_17_19.json', 'r') as f:
# with open('labeled_dataset/tags_17_191.json', 'r') as f:
  tags_dict = json.load(f)
def isSandwichAttacker(adderss):
  return (adderss.lower() in tags_dict['SandwichAttacker']) or (adderss.lower() in tags_dict['SandwichMiddleAttacker'])
def isCommonTrader(adderss):
  return adderss.lower() in tags_dict['CommonTrader']
# 标签地址
with open('labeled_dataset/tags_17_198.json', 'r') as f:
  tags_label_dict = json.load(f)
def CommonTraderLabel(adderss):
  return adderss.lower() in tags_label_dict['CommonTrader']
# isSandwichAttacker('0xc47ca739c179e8a51fe2713dd038cd87261cadfb')

# 获取地址发出交易的数量, 一般用于外部地址，合约地址没必要使用
def getTransactionCount(address):
  payload = json.dumps({
    "method": "eth_getTransactionCount",
    "params": [
      address,
      "latest"
    ],
    "id": 1,
    "jsonrpc": "2.0"
  })
  response = requests.request("POST", url, headers=headers, data=payload)
  if response.json()['result'] is not None:
    return int(response.json()['result'], 16)
# getTransactionCount('0xc47ca739c179e8a51fe2713dd038cd87261cadfb')

# 获取地址的内部代币交易频次
def getLogsCount(address, fromBlock, step):
  payload = json.dumps({
    "method": "eth_getLogs",
    "params": [
      {
        "address": address,
        "fromBlock": hex(fromBlock),
        "toBlock": hex(fromBlock+step),
      }
    ],
    "id": 1,
    "jsonrpc": "2.0"
  })
  response = requests.request("POST", url, headers=headers, data=payload)
  logs = response.json()['result']
  transfer_logs = [l for l in logs if ((len(l['topics'])==3) and l['topics'][0]=='0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef')]
  token_logs = [t['address'] for t in logs]
  return len(transfer_logs) / step, len(logs) / step # , set(token_logs)


'''通过输入的dict信息构建交易图

'''
def buildGraph(swtx, mode):
    if pd.isna(swtx['to']):
        return 
    # 节点标识
    node_dict = {swtx['from']: 0, swtx['to']: 1}
    node_idx = 2  

    # 读取所有日志，遍历
    swtx_logs_json = json.loads(swtx['logs'].replace("'", '"').replace('False','false'))

    if len(swtx_logs_json) == 0:
        return
    
    swtx_txMeta_json = json.loads(swtx['txMeta'].replace("'", '"').replace('False','false'))
    if 'gasPrice' in swtx_txMeta_json:
        gasPrice = swtx_txMeta_json['gasPrice'] / 10**9
    else:
        gasPrice = 0

    transfer_list = []  # 记录所有代币转移
    for l in swtx_logs_json:
        # 保留Transfer的Topic
        if(len(l['topics']) ==3 ):
            # 仅保留转移事件
            if l['topics'][0] == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                transfer = {}
                transfer['address'] = l['address']
                transfer['topics0'] = l['topics'][0]
                transfer['from'] = '0x'+l['topics'][1][-40:].lower()
                transfer['to'] = '0x'+l['topics'][2][-40:].lower()
                transfer['data'] = int(l['data'],16) / 10**18
                transfer['logIndex'] = int(l['logIndex'],16)
                transfer_list.append(transfer)
    
    # 排错
    if len(transfer_list) == 0:
        return 

    blockNumber = int(swtx['blockNumber'], 16)
    transactionIndex = int(swtx['transactionIndex'], 16)
    gasUsed = int(swtx['gasUsed'], 16) / 10**5

    # 针对代币转移事件进行节点编号
    for t in transfer_list:
        if not (t['from'] in node_dict):
            node_dict[t['from']] = node_idx
            node_idx += 1
        if not (t['to'] in node_dict):
            node_dict[t['to']] = node_idx
            node_idx += 1
    
    # 节点连线
    FromT = th.tensor([], dtype=th.int32)
    ToT = th.tensor([], dtype=th.int32)
    FromT = th.cat((FromT, th.tensor([0])), 0)
    ToT = th.cat((ToT, th.tensor([1])), 0)

    for t in transfer_list:
        FromT = th.cat((FromT, th.tensor([node_dict[t['from']]])), 0)
        ToT = th.cat((ToT, th.tensor([node_dict[t['to']]])), 0)

    graph = dgl.graph((FromT, ToT))

    # 添加特征
    # 节点: 0是否三明治攻击常客\1是否MEV-bot\2是否合约\34代币+其他交易频率\_5交易代币类型对\6是否常用交易所; 
    # 节点备选：\总收入\平均收入\最小收入\最大收入\总支出\平均支出\最小支出\最大支出\平均交易时间跨度\生命周期\余额
    
    node_list = list(node_dict.keys())
    # From是否三明治攻击常客 ✔ To是否Etherscan获取MEV-bot标签列表 ✔
    node_feature_array = np.zeros([node_idx, 7])
    # 交易的From和To
    node_feature_array[0][0] = int(isSandwichAttacker(node_list[0]))  # From
    node_feature_array[1][1] = int(isMEVBots(node_list[1]))  # To
    node_feature_array[1][2] = 1

    for nid, n in enumerate(node_list[2:]):
        node_feature_array[nid + 2][2] = int(isContract(n)) # 通过rpc判断是否合约 ✔
        node_feature_array[nid + 2][3], node_feature_array[nid + 2][4] = getLogsCount(n, blockNumber, 10) # 通过账户的历史交易信息，除以间隔时间Transfer/其他 ✔
        if mode == 'train':
            node_feature_array[nid + 2][6] = int(isCommonTrader(n)) # 通过交易所列表判断是否常用交易所 ✔

    # 在当前交易进出代币数量，TODO历史是否统计，同时统计边特征
    edge_feature_array = np.zeros([len(FromT), 6])

    # 边：0交易额\1时间戳or区块号\2logIndex\3gasUsed\4gasPrice\5是否是WETH

    # 外部交易
    edge_feature_array[0][0] = t['data']
    edge_feature_array[0][1] = blockNumber / 10**7  # 调整数量级, 区块号代替时间戳
    edge_feature_array[0][2] = transactionIndex
    edge_feature_array[0][3] = gasUsed
    edge_feature_array[0][4] = gasPrice
    # 内部交易
    token_dict = {}
    for tid, t in enumerate(transfer_list):
        if node_dict[t['from']] not in token_dict:
            token_dict[node_dict[t['from']]] = [t['address']]
        else:
            token_dict[node_dict[t['from']]].append(t['address'])
        if node_dict[t['to']] not in token_dict:
            token_dict[node_dict[t['to']]] = [t['address']]
        else:
            token_dict[node_dict[t['to']]].append(t['address'])
        edge_feature_array[tid+1][0] = t['data']
        edge_feature_array[tid+1][1] = blockNumber / 10**7  # 调整数量级, 区块号代替时间戳
        edge_feature_array[tid+1][2] = t['logIndex'] / 10
        # 是否是WETH
        if t['address'] == '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2':
            edge_feature_array[tid+1][5] = 1
    # 对数归一化
    edge_feature_array[:, 0] = np.log1p(edge_feature_array[:, 0])  # log(1 + x)

    # 代币数量
    for n, v in token_dict.items():
        node_feature_array[n][5] = len(set(v))

    graph.ndata['feature'] = th.tensor(node_feature_array)
    graph.edata['feature'] = th.tensor(edge_feature_array)
    node_label_array = np.zeros([node_idx])
    # 节点DEX地址标签
    for nid, n in enumerate(node_list):
        node_label_array[nid] = int(CommonTraderLabel(n))
    graph.ndata['label'] = th.tensor(node_label_array)
    graph = dgl.add_self_loop(graph)
    return graph


if __name__ == '__main__':
    data_df = pd.read_csv('HeuristicOutput/Graphs/1.csv')
    graphs = []
    graphs_labels = []
    for d in range(len(data_df)):
        graphs.append(buildGraph(data_df.iloc[d,:], 'test'))
        graphs_labels.append(1)

    dgl.save_graphs(f"HeuristicOutput/Graphs/1.dgl", graphs, {"glabels": th.tensor(graphs_labels)})