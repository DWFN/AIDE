import ctypes

import pandas as pd

import torch
import numpy as np
import os

import dgl
def get_node(pattern_path,cve_path,casual_path): #获得节点列表
    pattern = pd.read_excel(pattern_path)
    cve = pd.read_excel(cve_path)
    casual = pd.read_excel(casual_path)

# 获取所有技术名称
    node_set = set()
    node_set.update(pattern.columns)
    node_set.update(cve.columns)
    node_set.update(casual.columns)
    node_list = list(node_set)
    node_list.remove('Unnamed: 0')
    node_list.sort()
    tech_index = {tech: idx for idx, tech in enumerate(node_list)}
    return  tech_index
def build_graph(relations): #建立图
    num_nodes = 123
    edges = []
    weights = []

    for rel in relations:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if relations[rel][i, j] > 0:
                    edges.append((i, j))
                    weights.append(relations[rel][i, j])

    #创建 DGL 图
    src, dst = zip(*edges)
    g = dgl.graph((src, dst), num_nodes=num_nodes)

    # 将权重添加为边数据
    g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    return g
def get_graph(pattern_path,cve_path,casual_path): #获得图
    pattern = pd.read_excel(pattern_path)
    cve = pd.read_excel(cve_path)
    casual = pd.read_excel(casual_path)
    tech_index = get_node(pattern_path,cve_path,casual_path)
    causal_dependency_matrix = np.zeros((len(tech_index), len(tech_index)))
    pattern_dependency_matrix = np.zeros((len(tech_index), len(tech_index)))
    cve_dependency_matrix = np.zeros((len(tech_index), len(tech_index)))

    for i in range(len(pattern.columns) - 1):
        for j in range(len(pattern.columns)):
            if pattern.iloc[i, j] != str and pattern.iloc[i, j] != 0 and pattern.columns[j] != 'Unnamed: 0':
                pattern_dependency_matrix[tech_index[pattern.iloc[i, 0]]][tech_index[pattern.columns[j]]] = pattern.iloc[
                    i, j]
    for i in range(len(casual.columns) - 1):
        for j in range(len(casual.columns)):
            if casual.iloc[i, j] != str and casual.iloc[i, j] != 0 and casual.columns[j] != 'Unnamed: 0':
                causal_dependency_matrix[tech_index[casual.iloc[i, 0]]][tech_index[casual.columns[j]]] = casual.iloc[i, j]
    for i in range(len(cve.columns) - 1):
        for j in range(len(cve.columns)):
            if cve.iloc[i, j] != str and cve.iloc[i, j] != 0 and cve.columns[j] != 'Unnamed: 0':
                cve_dependency_matrix[tech_index[cve.iloc[i, 0]]][tech_index[cve.columns[j]]] = cve.iloc[i, j]

    # 定义关系矩阵 (假设已给出)
    relations = {
        'cve_dependency': cve_dependency_matrix,
        'pattern_dependency': pattern_dependency_matrix,
        'causal_dependency': causal_dependency_matrix
    }

# 构建图
    graph = build_graph(relations)
# graph = graph.to(device)
    print(graph)
    return  graph