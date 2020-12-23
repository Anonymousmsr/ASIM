# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified network structure

import torch
from .modules import Module, ModuleList, ModuleDict
from .modules.embedding import Embedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import Pooling
from .modules.prediction import registry as prediction
from .modules.encoder import Seq2SeqEncoder
import torch.nn as nn
import math

class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)
        

    def forward(self, x):
        return self.model(x)

class Network(Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding = Embedding(args)
        self.encoding = Seq2SeqEncoder(nn.LSTM,
                            args.embedding_dim,
                            args.hidden_size,
                            bidirectional=True)

        self.encoding_2 = Seq2SeqEncoder(nn.LSTM,
                            args.embedding_dim + args.hidden_size,
                            args.hidden_size,
                            bidirectional=True)

        self.alignment_inter = alignment[args.alignment](args, args.embedding_dim + args.hidden_size)
        self.fusion = fusion[args.fusion](args, args.embedding_dim + args.hidden_size * 2)
        self.connection = connection[args.connection]()
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, inputs):
        a = inputs['text1']
        b = inputs['text2']
        mask_a1 = inputs['mask1']
        mask_b1 = inputs['mask2']
        
        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b
        
        mask_a = (torch.sum(a, dim=-1) != 0).float()
        mask_b = (torch.sum(b, dim=-1) != 0).float()
        a_length = mask_a.sum(dim=-1).long()
        b_length = mask_b.sum(dim=-1).long()

        a_enc = self.encoding(a, a_length)
        b_enc = self.encoding(b, b_length) 

        a = torch.cat([a, a_enc], dim=-1)
        b = torch.cat([b, b_enc], dim=-1)
        align_a, align_b = self.alignment_inter(a, b, mask_a1, mask_b1)

        a = self.fusion(a, align_a)
        b = self.fusion(b, align_b)

        a = self.connection(a, res_a, 1)
        b = self.connection(b, res_b, 1)

        a_enc = self.encoding_2(a, a_length)
        b_enc = self.encoding_2(b, b_length)

        a = torch.cat([a, a_enc], dim=-1)
        b = torch.cat([b, b_enc], dim=-1)
    
        a = self.pooling(a, mask_a1)
        b = self.pooling(b, mask_b1)
        return self.prediction(a, b)
