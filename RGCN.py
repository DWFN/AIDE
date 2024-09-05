import ctypes

import torch

import torch.nn as nn
import torch.nn.functional as F
import os
from dgl.nn.pytorch import RelGraphConv
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels):
        super(RGCN, self).__init__()
        self.layer1 = RelGraphConv(num_nodes, 64, num_rels, regularizer='basis', num_bases=num_rels)
        self.layer2 = RelGraphConv(h_dim, 32, num_rels, regularizer='basis', num_bases=num_rels)
        # self.layer3 = RelGraphConv(32, out_dim, num_rels, regularizer='basis', num_bases=num_rels)

    def forward(self, g, feat, etypes):
        h = self.layer1(g, feat, etypes)
        h = F.relu(h)
        # h = self.dropout(h)  # 添加 Dropout
        h = self.layer2(g, h, etypes)
        h = F.relu(h)
        # h = self.layer3(g, h, etypes)
        # h = F.relu(h)
        # print(h)
        return h

class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels, transformer_dim, out_dim,
                 num_classes):
        super(Model, self).__init__()
        self.rgcn = RGCN(num_nodes, h_dim, out_dim, num_rels)
        self.pool = nn.AdaptiveMaxPool1d(1024)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=transformer_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.fc = nn.Linear(transformer_dim, num_classes)

    def forward(self, g, inputs, etypes):
        h = self.rgcn(g, inputs, etypes)
        h = h.view(1, -1)
        h = self.pool(h)  # to 1x1024
        h = h.unsqueeze(0)  # Add batch dimension for Transformer
        h = self.decoder(h, h)
        h = self.fc(h)
        h = h.squeeze(0)  # Remove batch dimension
        return h
