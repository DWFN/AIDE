import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import pandas as pd
import numpy as np
import time
import gc
from RGCN import RGCN,Model
from get_graph import get_graph
from get_data import encode_train_event
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()

    def forward(self, predictions, labels):
        mask = labels == 1
        predictions = predictions[mask]
        labels = labels[mask]
        loss = F.binary_cross_entropy(predictions, labels)
        return loss




def train_RGCN(num_nodes, h_dim, out_dim, num_rels, num_epochs, batch, pattern_path, cve_path, casual_path, data_path,
               model_name):
    model = Model(num_nodes=num_nodes, h_dim=h_dim, num_rels=num_rels,transformer_dim=1024,out_dim=out_dim,
                  num_classes=123)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = MaskedBCELoss().to(device)
    graph = get_graph(pattern_path, cve_path, casual_path)
    if (os.path.exists(model_name) == False):

        for epoch in range(num_epochs):
            start_time = time.time()  # 记录开始时间
            model.train()
            graph = graph.to(device)
            losses = float(0)
            # i = 0
            data = pd.read_csv(data_path, chunksize=batch)
            for tra in data:
                loss = 0

                train_matrices ,train_label_matrices = encode_train_event(tra)
                train_ma = np.array(train_matrices)
                train_label_ma = np.array(train_label_matrices)
                print(train_ma.shape)
                input = torch.tensor(train_ma, dtype=torch.float32).to(device)
                input_label = torch.tensor(train_label_ma, dtype=torch.float32).to(device)
                for event in range(len(train_ma)):
                    input_features = input[event]
                    logits = model(graph, input_features, graph.edata['weight'])
                    logits = F.sigmoid(logits)
                    logits = logits.view(logits.shape[1], 1)
                    labels_diag = torch.diag(input_label[event])
                    labels_diag =labels_diag.view(labels_diag.shape[0],1 )
                    loss += criterion(logits, labels_diag)

                losses += loss.item()
                print(losses)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()

            model_save = model_name + str(epoch)
            torch.save(model.state_dict(), model_save)
            end_time = time.time()  # 记录结束时间
            epoch_time = end_time - start_time  # 计算训练周期消耗时间
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses}, Time: {epoch_time:.2f} seconds")

        torch.save(model.state_dict(), model_name)

    else:
        model.load_state_dict(torch.load(model_name))

    return model, graph
