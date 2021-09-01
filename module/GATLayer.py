#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Yuqi Zhou"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,GC software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

######################################### SubLayer #########################################
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output


######################################### HierLayer #########################################

class SGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=0):
        super(SGATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        sedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 0)
        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=sedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]


class WSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])                  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        # print("edge e ", edges.data['e'].size())
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        g.nodes[wnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=wsedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]


class SWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=swedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]


class IGWWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["ig_weight_embed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        igwnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        igwwedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        g.nodes[igwnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=igwwedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]


class MulHeadAttentionDecoder(nn.Module):
    def __init__(self, hps, n_features, num_heads=8):
        # multi-head-attention
        super().__init__()
        self.batch_size = hps.batch_size
        self.n_feature = hps.hidden_size

        self.wd2sd = torch.nn.Linear(hps.word_emb_dim, self.n_feature)
        self.IG_mulatt   = torch.nn.MultiheadAttention(self.n_feature, num_heads=num_heads)
        self.HSG1_mulatt = torch.nn.MultiheadAttention(self.n_feature, num_heads=num_heads)
        self.HSG2_mulatt = torch.nn.MultiheadAttention(self.n_feature, num_heads=num_heads)

        self.ffn1 = PositionwiseFeedForward(n_features, hps.ffn_inner_hidden_size)
        self.ffn2 = PositionwiseFeedForward(n_features, hps.ffn_inner_hidden_size)
        self.ffn = PositionwiseFeedForward(n_features, hps.ffn_inner_hidden_size)

    def forward(self,iinpute, winpute, sinpute):
        winpute = self.wd2sd(winpute).unsqueeze(1)       # [wnodes, 1, self.n_feature]
        iinpute = self.wd2sd(iinpute).unsqueeze(1)       # [inodes, 1, self.n_feature]
        sinpute = sinpute.unsqueeze(1)                   # [snodes, 1, self.n_feature]

        # multi head attention
        iw_attention, _ = self.IG_mulatt(winpute, iinpute, iinpute)               # [wnodes, 1, self.n_feature]
        iw_attention = self.ffn1(iw_attention)
        sw_attention, _ = self.HSG1_mulatt(winpute, sinpute, sinpute)             # [wnodes, 1, self.n_feature]
        sw_attention = self.ffn2(sw_attention)
        s_attention, _  = self.HSG2_mulatt(sinpute, iw_attention, sw_attention) # [snodes, 1, self.n_feature]
        # s_attention, _  = self.HSG2_mulatt(sinpute, winpute, winpute)   # [snodes, 1, self.n_feature]

        # FFN + add + normalization
        # outputes = self.ffn(s_attention)
        outputes = self.ffn(s_attention)
        outputes = outputes.squeeze(1)     # [snodes, self.n_feature]
        return outputes

