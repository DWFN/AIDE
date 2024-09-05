
import torch

import pandas as pd
import numpy as np

from get_graph import get_node
from get_data import  get_tec
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


pattern = pd.read_excel('pattern_transition_matrix.xlsx')
cve = pd.read_excel('cve_transition_matrix.xlsx')
casual = pd.read_excel('causal_transition_matrix.xlsx')

# 获取所有技术名称
node_set = set()
node_set.update(pattern.columns)
node_set.update(cve.columns)
node_set.update(casual.columns)
node_list = list(node_set)
node_list.remove('Unnamed: 0')
node_list.sort()
tech_index = {tech: idx for idx, tech in enumerate(node_list)}
def encode_test_event(data,pattern_path,cve_path,casual_path):  #获得测试数据
    tech_index = get_node(pattern_path,cve_path,casual_path)
    test_2_matrices = []
    test_3_matrices = []
    test_4_matrices = []
    test_5_matrices = []
    test_6_matrices = []
    test_matrices = []
    for _, event in data.iterrows():
        count = 0
        event_matrix = np.zeros((123, 123))
        event_2_matrix = np.zeros((123, 123))
        event_3_matrix = np.zeros((123, 123))
        event_4_matrix = np.zeros((123, 123))
        event_5_matrix = np.zeros((123, 123))
        event_6_matrix = np.zeros((123, 123))
        techniques_in_event = event['tid'].split(',')  # 假设技术列名为'techniques'
        for i in range(len(techniques_in_event)):
            techniques_in_event[i] = get_tec(techniques_in_event[i])
        for tech in techniques_in_event:
            if tech in tech_index.keys():
                idx = tech_index[tech]
                if event_matrix[idx, idx] != 1:
                    event_matrix[idx, idx] = 1
                    if count < 2:
                        event_2_matrix[idx, idx] = 1
                        event_3_matrix[idx, idx] = 1
                        event_4_matrix[idx, idx] = 1
                        event_5_matrix[idx, idx] = 1
                        event_6_matrix[idx, idx] = 1
                    elif count < 3:
                        event_3_matrix[idx, idx] = 1
                        event_4_matrix[idx, idx] = 1
                        event_5_matrix[idx, idx] = 1
                        event_6_matrix[idx, idx] = 1
                    elif count < 4:
                        event_4_matrix[idx, idx] = 1
                        event_5_matrix[idx, idx] = 1
                        event_6_matrix[idx, idx] = 1
                    elif count < 5:
                        event_5_matrix[idx, idx] = 1
                        event_6_matrix[idx, idx] = 1
                    elif count < 6:
                        event_6_matrix[idx, idx] = 1
                    count += 1
        if (count == 9):
            test_matrices.append(event_matrix)
            test_2_matrices.append(event_2_matrix)
            test_3_matrices.append(event_3_matrix)
            test_4_matrices.append(event_4_matrix)
            test_5_matrices.append(event_5_matrix)
            test_6_matrices.append(event_6_matrix)
        else:
            continue

    return test_matrices, test_2_matrices, test_3_matrices, test_4_matrices, test_5_matrices, test_6_matrices

def get_test_data(data): #获得训练矩阵
    test_matrices = []
    test_label_matrices = []
    for _, event in data.iterrows():
        count = 0
        event_matrix = np.zeros((123, 123))
        event_label_matrix = np.zeros((123, 123))
        techniques_in = event['tid'].split(',')
        techniques_in_event = list()
        for i in range(len(techniques_in) - 1):
            techniques_in_event.append(get_tec(techniques_in[i]))
        event_tec_list = list()
        for tech in techniques_in_event:
            if tech in tech_index.keys():
                if (tech in event_tec_list) == False:
                    event_tec_list.append(tech)

        if (len(event_tec_list) == 0 or len(event_tec_list) == 1):
            continue
        for tech in techniques_in_event:
            idx = tech_index[tech]
            if count == len(techniques_in_event) - 1:
                event_label_matrix[idx, idx] = 1
            else:
                event_matrix[idx, idx] = 1
            count += 1

        test_matrices.append(event_matrix)
        test_label_matrices.append(event_label_matrix)

    return test_matrices ,test_label_matrices

def get_test_ByLength_data(data,length):
    test_matrices = []
    test_label_matrices = []
    for _, event in data.iterrows():
        count = 0
        event_matrix = np.zeros((123, 123))
        event_label_matrix = np.zeros((123, 123))
        techniques_in = event['tid'].split(',')
        techniques_in_event = list()
        for i in range(len(techniques_in) - 1):
            techniques_in_event.append(get_tec(techniques_in[i]))
        event_tec_list = list()
        for tech in techniques_in_event:
            if tech in tech_index.keys():
                if (tech in event_tec_list) == False:
                    event_tec_list.append(tech)

        if (len(event_tec_list) != length):
            continue
        for tech in techniques_in_event:
            idx = tech_index[tech]
            if count > 1:
                event_label_matrix[idx, idx] = 1
            else:
                event_matrix[idx, idx] = 1
            count += 1

        test_matrices.append(event_matrix)
        test_label_matrices.append(event_label_matrix)

    return test_matrices ,test_label_matrices

def get_test_ByLength_in2(data):
    test_matrices = []
    test_label_matrices = []
    for _, event in data.iterrows():
        count = 0
        event_matrix = np.zeros((123, 123))
        event_label_matrix = np.zeros((123, 123))
        techniques_in = event['tid'].split(',')
        techniques_in_event = list()
        for i in range(len(techniques_in) - 1):
            techniques_in_event.append(get_tec(techniques_in[i]))
        event_tec_list = list()
        for tech in techniques_in_event:
            if tech in tech_index.keys():
                if (tech in event_tec_list) == False:
                    event_tec_list.append(tech)

        for tech in techniques_in_event:
            idx = tech_index[tech]
            if count > 1:
                event_label_matrix[idx, idx] = 1
            else:
                event_matrix[idx, idx] = 1
            count += 1

        test_matrices.append(event_matrix)
        test_label_matrices.append(event_label_matrix)

    return test_matrices ,test_label_matrices

def test_RGCN(model,graph,test_path):
    model.eval()
    data = pd.read_csv(test_path)
    test_matrices, test_label_matrices = get_test_data(data)
    print("test_len:",len(test_matrices))
    ac1 = 0
    ac3 = 0
    ac5 = 0
    model = model.to(device)
    graph = graph.to(device)
    for i in range(len(test_matrices)):
        input_features = torch.tensor(test_matrices[i], dtype=torch.float32).to(device)  # 使用测试矩阵作为输入特征
        with torch.no_grad():
            logits = model(graph, input_features, graph.edata['weight'])
            # logits = model(input_features)
        _, top_indices_1 = torch.topk(logits, 1, dim=1)
        _, top_indices_3 = torch.topk(logits, 3, dim=1)
        _, top_indices_5 = torch.topk(logits, 5, dim=1)
        labels = torch.tensor(test_label_matrices[i], dtype=torch.float32).to(device)
        labels_diag = torch.diag(labels)
        for i in range(top_indices_1.shape[1]):
            if(labels_diag[top_indices_1[0][i]]==1):
                ac1 += 1
                break
        for i in range(top_indices_5.shape[1]):
            if(labels_diag[top_indices_5[0][i]]==1):
                ac5 += 1
                break
        for i in range(top_indices_3.shape[1]):
            if(labels_diag[top_indices_3[0][i]]==1):
                ac3 += 1
                break

    print("1:",ac1/len(test_matrices))
    print("3:",ac3/len(test_matrices))
    print("5:",ac5/len(test_matrices))

def test_RGCN_hit_ByLength(model,graph,test_path):
    model.eval()
    data = pd.read_csv(test_path)
    test_matrices, test_label_matrices = get_test_data(data)
    print("test_len:",len(test_matrices))
    hit1 = np.zeros(123)
    hit3 = np.zeros(123)
    hit5 = np.zeros(123)
    length_list = np.zeros(123)
    model = model.to(device)
    graph = graph.to(device)
    for i in range(len(test_matrices)):
        input_features = torch.tensor(test_matrices[i], dtype=torch.float32).to(device)  # 使用测试矩阵作为输入特征
        length = (input_features > 0).sum().item() +1
        print(length)
        length_list[length] += 1
        with torch.no_grad():
            logits = model(graph, input_features, graph.edata['weight'])
        _, top_indices_1 = torch.topk(logits, 1, dim=1)
        _, top_indices_3 = torch.topk(logits, 3, dim=1)
        _, top_indices_5 = torch.topk(logits, 5, dim=1) 
        labels = torch.tensor(test_label_matrices[i], dtype=torch.float32).to(device)
        labels_diag = torch.diag(labels)
        for i in range(top_indices_1.shape[1]):
            if(labels_diag[top_indices_1[0][i]]==1):
                hit1[length] += 1
                break
        for i in range(top_indices_3.shape[1]):
            if(labels_diag[top_indices_3[0][i]]==1):
                hit3[length] += 1
                break
        for i in range(top_indices_5.shape[1]):
            if(labels_diag[top_indices_5[0][i]]==1):
                hit5[length] += 1
                break
    for i in range(len(length_list)):
        if(length_list[i] >0):
            print("length_",i," hit1:",hit1[i]/length_list[i])
            print("length_",i," hit3:",hit3[i]/length_list[i])
            print("length_",i," hit5:",hit5[i]/length_list[i])

def test_RGCN_map(model,graph,test_path):
    model.eval()
    data = pd.read_csv(test_path)
    test_matrices, test_label_matrices = get_test_data(data)
    print("test_len:",len(test_matrices))
    # hit1 = np.zeros(123)
    # hit3 = np.zeros(123)
    # length_list = np.zeros(123)
    model = model.to(device)
    graph = graph.to(device)
    all_ap1 = 0
    all_ap3 = 0
    all_ap5 = 0
    for i in range(len(test_matrices)):
        input_features = torch.tensor(test_matrices[i], dtype=torch.float32).to(device)  # 使用测试矩阵作为输入特征
        R1 = 0
        R3 = 0
        R5 = 0
        ap1 = 0
        ap3 = 0
        ap5 = 0
        with torch.no_grad():
            logits = model(graph, input_features, graph.edata['weight'])
            # logits = model(input_features)
        _, top_indices_1 = torch.topk(logits, 1, dim=1) 
        _, top_indices_3 = torch.topk(logits, 3, dim=1) 
        _, top_indices_5 = torch.topk(logits, 5, dim=1) 
        labels = torch.tensor(test_label_matrices[i], dtype=torch.float32).to(device)
        labels_diag = torch.diag(labels)
        for i in range(top_indices_1.shape[1]):
            if(labels_diag[top_indices_1[0][i]]==1):
                R1 += 1
                ap1 += R1/(i+1)

        for i in range(top_indices_3.shape[1]):
            if(labels_diag[top_indices_3[0][i]]==1):
                R3 += 1
                ap3 += R3/(i+1)
        for i in range(top_indices_5.shape[1]):
            if(labels_diag[top_indices_5[0][i]]==1):
                R5 += 1
                ap5 += R5/(i+1)
        if(R1>0):
            all_ap1 += ap1/R1
        if(R3>0):
            all_ap3 += ap3/R3
        if(R5>0):
            all_ap5 += ap5/R5
    print("Map1:",all_ap1/len(test_matrices))
    print("Map3:",all_ap3/len(test_matrices))
    print("Map5:",all_ap5/len(test_matrices))

def test_RGCN_map_ByLength(model,graph,test_path):
    model.eval()
    data = pd.read_csv(test_path)
    test_matrices, test_label_matrices = get_test_data(data)
    print("test_len:",len(test_matrices))
    all_ap1 = np.zeros(123)
    all_ap3 = np.zeros(123)
    all_ap5 = np.zeros(123)
    length_list = np.zeros(123)
    model = model.to(device)
    graph = graph.to(device)
    for i in range(len(test_matrices)):
        input_features = torch.tensor(test_matrices[i], dtype=torch.float32).to(device)  # 使用测试矩阵作为输入特征
        length = (input_features > 0).sum().item() +1
        print(length)
        length_list[length] += 1
        R1 = 0
        R3 = 0
        R5 = 0
        ap1 = 0
        ap3 = 0
        ap5 = 0
        with torch.no_grad():
            logits = model(graph, input_features, graph.edata['weight'])
        _, top_indices_1 = torch.topk(logits, 1, dim=1)
        _, top_indices_3 = torch.topk(logits, 3, dim=1)  
        _, top_indices_5 = torch.topk(logits, 5, dim=1)  
        labels = torch.tensor(test_label_matrices[i], dtype=torch.float32).to(device)
        labels_diag = torch.diag(labels)
        for i in range(top_indices_1.shape[1]):
            if(labels_diag[top_indices_1[0][i]]==1):
                R1 += 1
                ap1 += R1/(i+1)
        for i in range(top_indices_3.shape[1]):
            if(labels_diag[top_indices_3[0][i]]==1):
                R3 += 1
                ap3 += R3/(i+1)
        for i in range(top_indices_5.shape[1]):
            if(labels_diag[top_indices_5[0][i]]==1):
                R5 += 1
                ap5 += R5/(i+1)
        if(R1>0):
            all_ap1[length] += ap1/R1
        if(R3>0):
            all_ap3[length] += ap3/R3
        if(R5>0):
            all_ap5[length] += ap5/R5
    for i in range(len(length_list)):
        if(length_list[i] >0):
            print("length_",i," map1:",all_ap1[i]/length_list[i])
            print("length_",i," map3:",all_ap3[i]/length_list[i])
            print("length_",i," map5:",all_ap5[i]/length_list[i])

def test_RGCN_Recall_ByLength_in2(model,graph,test_path,length):
    model.eval()
    data = pd.read_csv(test_path)
    test_matrices, test_label_matrices = get_test_ByLength_data(data,length)
    print("test_len:",len(test_matrices))
    Recall_5 = 0
    Recall_10 = 0
    Recall_15 = 0
    Recall_20 = 0
    model = model.to(device)
    graph = graph.to(device)
    for i in range(len(test_matrices)):
        hit5 = 0
        hit10 =0
        hit15 = 0
        hit20 = 0
        input_features = torch.tensor(test_matrices[i], dtype=torch.float32).to(device)  # 使用测试矩阵作为输入特征
        with torch.no_grad():
            logits = model(graph, input_features, graph.edata['weight'])
        _, top_indices_5 = torch.topk(logits, 5, dim=1)  
        _, top_indices_10 = torch.topk(logits, 10, dim=1)  
        _, top_indices_15 = torch.topk(logits, 15, dim=1)  
        _, top_indices_20 = torch.topk(logits, 20, dim=1)  
        labels = torch.tensor(test_label_matrices[i], dtype=torch.float32).to(device)
        length = (labels > 0).sum().item()
        #print(length)
        labels_diag = torch.diag(labels)
        for i in range(top_indices_5.shape[1]):
            if(labels_diag[top_indices_5[0][i]]==1):
                hit5 += 1
        for i in range(top_indices_10.shape[1]):
            if(labels_diag[top_indices_10[0][i]]==1):
                hit10 += 1
        for i in range(top_indices_15.shape[1]):
            if(labels_diag[top_indices_15[0][i]]==1):
                hit15 += 1
        for i in range(top_indices_20.shape[1]):
            if(labels_diag[top_indices_20[0][i]]==1):
                hit20 += 1
        Recall_5 += hit5 / length
        Recall_10 += hit10 / length
        Recall_15 += hit15 / length
        Recall_20 += hit20 / length
    print(" Recall5:",Recall_5/len(test_matrices))
    print(" Recall10:",Recall_10/len(test_matrices))
    print(" Recall15:",Recall_15/len(test_matrices))
    print(" Recall20:",Recall_20/len(test_matrices))

def test_RGCN_Recall_in2(model,graph,test_path):
    model.eval()
    data = pd.read_csv(test_path)
    test_matrices, test_label_matrices = get_test_ByLength_in2(data)
    print("test_len:",len(test_matrices))
    Recall_5 = 0
    Recall_10 = 0
    Recall_15 = 0
    Recall_20 = 0
    model = model.to(device)
    graph = graph.to(device)
    for i in range(len(test_matrices)):
        hit5 = 0
        hit10 =0
        hit15 = 0
        hit20 = 0
        input_features = torch.tensor(test_matrices[i], dtype=torch.float32).to(device)  # 使用测试矩阵作为输入特征
        with torch.no_grad():
            logits = model(graph, input_features, graph.edata['weight'])
            #logits = model(input_features)
        _, top_indices_5 = torch.topk(logits, 5, dim=1)  
        _, top_indices_10 = torch.topk(logits, 10, dim=1)  
        _, top_indices_15 = torch.topk(logits, 15, dim=1)  
        _, top_indices_20 = torch.topk(logits, 20, dim=1)  
        labels = torch.tensor(test_label_matrices[i], dtype=torch.float32).to(device)
        length = (labels > 0).sum().item()
        #print(length)
        labels_diag = torch.diag(labels)
        for i in range(top_indices_5.shape[1]):
            if(labels_diag[top_indices_5[0][i]]==1):
                hit5 += 1
        for i in range(top_indices_10.shape[1]):
            if(labels_diag[top_indices_10[0][i]]==1):
                hit10 += 1
        for i in range(top_indices_15.shape[1]):
            if(labels_diag[top_indices_15[0][i]]==1):
                hit15 += 1
        for i in range(top_indices_20.shape[1]):
            if(labels_diag[top_indices_20[0][i]]==1):
                hit20 += 1
        if(length != 0 ):
            Recall_5 += hit5 / length
            Recall_10 += hit10 / length
            Recall_15 += hit15 / length
            Recall_20 += hit20 / length
    print(" Recall5:",Recall_5/len(test_matrices))
    print(" Recall10:",Recall_10/len(test_matrices))
    print(" Recall15:",Recall_15/len(test_matrices))
    print(" Recall20:",Recall_20/len(test_matrices))
def test_RGCN_map_ByLength_in2(model,graph,test_path,length):
    model.eval()
    data = pd.read_csv(test_path)
    test_matrices, test_label_matrices = get_test_ByLength_data(data,length)
    print("test_len:",len(test_matrices))
    all_ap5 = 0
    all_ap10 = 0
    all_ap15 = 0
    all_ap20 = 0
    model = model.to(device)
    graph = graph.to(device)
    for i in range(len(test_matrices)):
        ap5 = 0
        ap10 = 0
        ap15 = 0
        ap20 = 0
        R5 = 0
        R10 = 0
        R15 = 0
        R20 = 0
        input_features = torch.tensor(test_matrices[i], dtype=torch.float32).to(device)  # 使用测试矩阵作为输入特征
        with torch.no_grad():
            logits = model(graph, input_features, graph.edata['weight'])
        _, top_indices_5 = torch.topk(logits, 5, dim=1)
        _, top_indices_10 = torch.topk(logits, 10, dim=1)  
        _, top_indices_15 = torch.topk(logits, 15, dim=1)  
        _, top_indices_20 = torch.topk(logits, 20, dim=1)  
        labels = torch.tensor(test_label_matrices[i], dtype=torch.float32).to(device)
        length = (labels > 0).sum().item() +1
        labels_diag = torch.diag(labels)
        for i in range(top_indices_5.shape[1]):
            if(labels_diag[top_indices_5[0][i]]==1):
                R5 += 1
                ap5 += R5/(i+1)
        for i in range(top_indices_10.shape[1]):
            if(labels_diag[top_indices_10[0][i]]==1):
                R10 += 1
                ap10 += R10/(i+1)
        for i in range(top_indices_15.shape[1]):
            if(labels_diag[top_indices_15[0][i]]==1):
                R15 += 1
                ap15 += R15/(i+1)
        for i in range(top_indices_20.shape[1]):
            if(labels_diag[top_indices_20[0][i]]==1):
                R20 += 1
                ap20 += R20/(i+1)
        if(R5>0):
            all_ap5 += ap5/R5
        if(R10>0):
            all_ap10 += ap10/R10
        if (R15 > 0):
            all_ap15 += ap15/R15
        if (R20 > 0):
            all_ap20 += ap20/R20
    print(" Map5:",all_ap5/len(test_matrices))
    print(" Map10:",all_ap10/len(test_matrices))
    print(" Map15:",all_ap15/len(test_matrices))
    print(" Map20:",all_ap20/len(test_matrices))