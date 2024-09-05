# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

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


def get_disorg(filename):  # 打乱指定文件
    if (os.path.exists('data_disorg.csv') == False):
        data = pd.read_csv(filename)  # 打乱数据
        data = shuffle(data)  # 打乱
        data = data.dropna(axis=0, how='any')  # 删除有缺失值的行
        data.to_csv('data_disorg.csv')


def get_data_conut(filename):  # 获得单个事件内技术数量
    data = pd.read_csv(filename)
    event_count = np.zeros(123)
    for _, event in data.iterrows():
        techniques_in_event = event['tid'].split(',')
        for i in range(len(techniques_in_event)-1):
            techniques_in_event[i] = get_tec(techniques_in_event[i])
        event_count[len(techniques_in_event)-1] += 1
    return event_count


def set_data(filename, out_data):  # 获得训练和测试csv
    data = pd.read_csv(filename)
    csv_data = pd.DataFrame(columns=['Unnamed: 0', 'tid', 'sighting_date'])
    for _, event in data.iterrows():
        techniques_in_event = event['tid'].split(',')  # 假设技术列名为'techniques'
        for i in range(len(techniques_in_event)):
            techniques_in_event[i] = get_tec(techniques_in_event[i])
        event_tec_list = list()
        for tech in techniques_in_event:
            if tech in tech_index.keys():
                flag = True
                for te in range(len(event_tec_list)):
                    if (event_tec_list[te] == tech):
                        flag = False
                        break
                if (flag == True):
                    event_tec_list.append(tech)
        csv_id = str()
        if (len(event_tec_list) == 1 or len(event_tec_list) == 0):
            continue
        for te in event_tec_list:
            csv_id += te + ","
        input = {'Unnamed: 0': event['Unnamed: 0'], 'tid': csv_id, 'sighting_date': event['sighting_date']}
        csv_data.loc[len(csv_data)] = input
        print(len(csv_data))
    csv_data.to_csv(out_data)


def train_test_csv(filename):
    data = pd.read_csv(filename)
    all_2_list = list()
    i=0
    for _, event in data.iterrows():
        str1 = event['tid']
        # i += 1
        # print(i)
        if len(str1.split(",")) -1 ==2:
            if (str1 in all_2_list)==False :
                all_2_list.append(str1)
    count_all_2 = np.zeros(len(all_2_list)) #有","
    print(all_2_list)
    print(len(all_2_list))
    cc = 0
    for i in range(len(all_2_list)):
        this_2 = all_2_list[i].split(",")
        cc+=1
        print(cc)
        has_2 = 0
        for _, event in data.iterrows():
            if len(event['tid'].split(',')) -1 !=2:
                for j in range(len(this_2)-1):
                    if event['tid'].find(this_2[j]) != -1:
                        has_2 += 1
                if(has_2==2):
                    count_all_2[i] += 1
            has_2 = 0
        print(count_all_2[i])
    print("2_list:",len(all_2_list),"have:",all_2_list)
    indices = np.argsort(count_all_2)[-len(count_all_2)//2:]
    print(indices)
    new_csv = pd.DataFrame(columns=['Unnamed: 0', 'tid', 'sighting_date'])
    locte = 0
    for _, event in data.iterrows():
        locte +=1
        f = True
        for del_2 in indices :
            if(event["tid"] == all_2_list[del_2]):
                f = False
                print("delete:",event["tid"],"event:",count_all_2[del_2],"locate:",locte,"len_csv:",len(new_csv))
                break


        if(f == True):
            input = {'Unnamed: 0': event['Unnamed: 0'], 'tid': event['tid'], 'sighting_date': event['sighting_date']}
            new_csv.loc[len(new_csv)] = input
    print(count_all_2)

def get_train_test_csv(filename,out_train,out_test):
    data = pd.read_csv(filename)
    event_count = get_data_conut(filename)
    st_test = np.zeros(123)
    print(event_count)
    # print(data.columns.tolist())
    train_scv = pd.DataFrame(columns=['Unnamed: 0', 'tid', 'sighting_date'])
    test_scv = pd.DataFrame(columns=['Unnamed: 0', 'tid', 'sighting_date'])
    for _, event in data.iterrows():
        techniques_in_event = event['tid'].split(',')  # 假设技术列名为'techniques'
        input = {'Unnamed: 0': event['Unnamed: 0'], 'tid': event['tid'], 'sighting_date': event['sighting_date']}

        if (st_test[len(techniques_in_event)-1] > event_count[len(techniques_in_event)-1] * 0.8):
            test_scv.loc[len(test_scv)] = input
            st_test[len(techniques_in_event)-1] += 1
        else:
            train_scv.loc[len(train_scv)] = input
            st_test[len(techniques_in_event)-1] += 1
        print(len(train_scv))
    test_scv.to_csv(out_test)
    train_scv.to_csv(out_train)


def get_tec(str1):
    str1 = str1.replace('[', "")
    str1 = str1.replace(',', "")
    str1 = str1.replace(']', "")
    str1 = str1.replace('\'', "")
    str1 = str1.replace(' ', "")
    return str1


def encode_train_event(data):  # 获得训练矩阵
    train_matrices = []
    train_label_matrices = []
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

        train_matrices.append(event_matrix)
        train_label_matrices.append(event_label_matrix)

    return train_matrices, train_label_matrices
