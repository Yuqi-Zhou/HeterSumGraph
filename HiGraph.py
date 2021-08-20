#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

# import torch      # 我自己注释掉的
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.GATLayer import MulHeadAttentionDecoder
from module.PositionEmbedding import get_sinusoid_encoding_table
from my_else import print_variable_info, my_stop, my_dividing_line    # ----------------------------------- 我自己加的


class HSumGraph(nn.Module):
    """ without sent2sent and add residual connection . sn 表示句子节点, wn 表示词节点"""
    def __init__(self, hps, embed):
        """

        :param hps:
        :param embed: word embedding
        """
        super().__init__()
        self._hps = hps  # 保存传入的参数 hps
        self._n_iter = hps.n_iter   # 迭代次数, 默认为1
        self._embed = embed  # 保存传入的词向量表 embed
        self.embed_size = hps.word_emb_dim  # 词向量长度, 默认300
        # sent node feature
        self._init_sn_param()   # 初始化句子节点的参数

        # _TFembed 是一个Embeding, 可以把编号映射成向量. 查找表可以映射10个, 是因为 tfidf*9取整数 只能是0~9, 所以10个刚刚好
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10  # feat_embed_size 默认为 50

        # #################################################↓↓↓↓↓↓    # ----------------------------------- 下面是我自己加的
        # 不可以将下面这行代码移到 set_igwnfeature 函数里, 否则的话会造成张量存放在 cpu 上而非 gpu 上
        # 查找表可以映射10个, 是因为 ig_weight * 9 取整数只能是0~9, 所以10个刚刚好
        self.ig_weight_embed = nn.Embedding(10, hps.feat_embed_size)  # feat_embed_size 默认为 50
        # #################################################↑↑↑↑↑↑    # ----------------------------------- 上面是我自己加的

        # self.n_feature_proj = nn.Linear(128 * 2, 64, bias=False)
        # n_feature_proj 可以将[xxxx, 256] 转换为 [xxxx, 64], 仅用于 forward 函数(不考虑多文档)
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(in_dim=embed_size,  # 300
                                out_dim=hps.hidden_size,    # 64
                                num_heads=hps.n_head,   # 8
                                attn_drop_out=hps.atten_dropout_prob,   # 0.1
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,  # 512
                                ffn_drop_out=hps.ffn_dropout_prob,  # 0.1
                                feat_embed_size=hps.feat_embed_size,    # 50
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,  # 64
                                out_dim=embed_size,     # 300
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,   # 0.1
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,  # 512
                                ffn_drop_out=hps.ffn_dropout_prob,  # 0.1
                                feat_embed_size=hps.feat_embed_size,    # 50
                                layerType="S2W"
                                )

        # #################################################↓↓↓↓↓↓    # ----------------------------------- 下面是我自己加的
        self.igword2word = WSWGAT(# in_dim=hps.hidden_size,  # 64
                                in_dim=embed_size,  # 300
                                out_dim=embed_size,  # 300
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,  # 0.1
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,  # 512
                                ffn_drop_out=hps.ffn_dropout_prob,  # 0.1
                                feat_embed_size=hps.feat_embed_size,  # 50
                                layerType="IGW2W"
                                )
        # #################################################↑↑↑↑↑↑    # ----------------------------------- 上面是我自己加的

        # node classification
        self.n_feature = hps.hidden_size    # 默认 64, 仅用于下一行(多文档的初始化中也有一行用到 self.n_feature)

        # sentence selector
        self.decoder = MulHeadAttentionDecoder(self._hps, self.n_feature)
        self.wh = nn.Linear(self.n_feature * 2, 2)  # self.wh = nn.Linear(64, 2)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :return: result: [sentnum, 2]
        """

        # 词节点初始化
        word_feature = self.set_wnfeature(graph)    # 得到所有单词节点的词向量  # [词节点数量, 300] [wnode, embed_size]
        # 句子节点初始化  # 原注释为: [snode, n_feature_size], 不对, 实际应为: [snode, hidden_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # 得到所有句子节点的句子特征向量, [句子数量, 64]
        # the start state   得到词的开始状态
        word_state = word_feature  # [词节点数量, 300]

        if self._hps.use_interest:
            igword_state = self.set_igwnfeature(graph)

            # igword -> word
            word_state = self.igword2word(graph, word_state, igword_state)

        # sent_state = self.word2sent(graph, word_feature, sent_feature)  # [句子数量, 64] 调用的是 WSWGAT 类的 forward 函数(即使是同样的文章, 多次运行得到的sent_state也不一样)
        sent_state = self.word2sent(graph, word_state, sent_feature)     # ----------------------------------- 我自己修改上一行的

        for i in range(self._n_iter):   # 默认 _n_iter 为1, 故 i==0
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)  # 调用的是 WSWGAT 类的 forward 函数

            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)  # 调用的是 WSWGAT 类的 forward 函数
        # sent_state 为 [句子数量, 64], result 为 [句子数量, 2]
        if self._hps.use_interest:
            sent_state2 = self.decoder(igword_state, word_state, sent_state)
            sent_state = torch.cat((sent_state, sent_state2), dim=1)
        result = self.wh(sent_state)

        return result

    def _init_sn_param(self):   # 只用于 __init__ 函数
        """
        本函数用于初始化句子节点的各种参数 -------------------- 我自己加的
        :return: 无
        """
        # self.sent_pos_embed = nn.Embedding.from_pretrained(51*300的<class 'torch.Tensor'> 类, freeze=True)
        # 得到 self.sent_pos_embed, 给函数 _sent_cnn_feature 使用
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        # self.cnn_proj = nn.Linear(300, 128)
        # 得到 self.cnn_proj , 给函数 _sent_cnn_feature 使用
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state    # 为 128
        # self.lstm = nn.LSTM(300, 128, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
        # 得到 self.lstm, 给函数 _sent_lstm_feature 使用
        # self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
        #                     batch_first=True, bidirectional=self._hps.bidirectional)
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers,
                            dropout=self._hps.recurrent_dropout_prob, batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:  # 默认为 True
            # self.lstm_proj = nn.Linear(128 * 2, 128)
            # 得到 self.lstm_proj, 给函数 _sent_lstm_feature 使用
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)
        # 得到 self.ngram_enc, 给函数 _sent_cnn_feature 使用
        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):   # 只用于 set_snfeature 函数
        # sentEncoder.forword(句子数量 * [本句的词id列表, 长度100, 补充0])
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # 得到所有句子的句向量, [snode, embed_size]

        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature    # 给所有句子节点加特征'sent_embedding', 值为句向量, [长度300]

        # 得到所有句子在文章中的位置, [1, 2, ..., 37(第1篇文章有37句), 1, 2, ...14(第2篇文章有14句), ...以此类推 ]  # [n_nodes]
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)
        position_embedding = self.sent_pos_embed(snode_pos)  # 得到所有句子的位置向量

        # 由句向量和位置向量得到 cnn_feature
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)   # cnn_proj = nn.Linear(300, 128)
        return cnn_feature  # [句子数量, 128]

    def _sent_lstm_feature(self, features, glen):   # 只用于 set_snfeature 函数
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))  # [n_nodes, n_feature_size] # self.lstm_proj = nn.Linear(128 * 2, 128)
        return lstm_feature  # [句子数量, 128]

    def set_wnfeature(self, graph):   # 只用于 forward 函数, 设置单词节点的特征 embed 和词句边的特征 tfidfembed, 并返回所有单词节点的词向量
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)  # 得到所有单词节点的 id
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # 得到所有词句边的 id
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]    # 得到所有单词节点的词 id
        w_embed = self._embed(wid)  # [n_wnodes, D]    # 得到所有单词节点的词向量
        graph.nodes[wnode_id].data["embed"] = w_embed   # 给所有单词节点加特征'embed', 值为对应的词向量
        etf = graph.edges[wsedge_id].data["tffrac"]     # 得到所有词句边的 tffrac
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)  # 给所有的词句边加特征'tfidfembed', 值为'tffrac'对应的50维向量
        return w_embed

    # #################################################↓↓↓↓↓↓    # ----------------------------------- 下面是我自己加的
    def set_igwnfeature(self, graph):
        igwnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 2)  # 得到所有兴趣图谱单词节点的 id
        igwid = graph.nodes[igwnode_id].data["igwid"]  # 得到所有兴趣图谱单词节点的词 id
        igw_embed = self._embed(igwid)  # 得到所有兴趣图谱单词节点的词向量
        graph.nodes[igwnode_id].data["igw_embed"] = igw_embed   # 给所有兴趣图谱单词节点加特征'igw_embed', 值为对应的词向量

        w2igw_edge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 2)  # 得到所有由单词节点指向兴趣图谱单词节点的边的id
        w2igw_ig_weight = graph.edges[w2igw_edge_id].data["ig_weight"]  # 得到这种边的 ig_weight
        graph.edges[w2igw_edge_id].data["ig_weight_embed"] = self.ig_weight_embed(w2igw_ig_weight)  # 给所有的这种边加特征'ig_weight_embed', 值为'ig_weight'对应的50维向量

        igw2w_edge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 3)  # 得到所有由兴趣图谱单词节点指向单词节点的边的id
        igw2w_ig_weight = graph.edges[igw2w_edge_id].data["ig_weight"]
        graph.edges[igw2w_edge_id].data["ig_weight_embed"] = self.ig_weight_embed(igw2w_ig_weight)

        # 下面这三行实际上没有作用, 因为没有用单词节点更新单词节点
        w2w_edge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 4)    # 得到所有由单词节点指向单词节点的边的id
        w2w_ig_weight = graph.edges[w2w_edge_id].data["ig_weight"]
        graph.edges[w2w_edge_id].data["ig_weight_embed"] = self.ig_weight_embed(w2w_ig_weight)

        return igw_embed
    # #################################################↑↑↑↑↑↑    # ----------------------------------- 上面是我自己加的

    def set_snfeature(self, graph):   # 只用于 forward 函数, 设置句子节点的特征
        # 得到 cnn_feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)   # 得到所有句子节点的 id
        cnn_feature = self._sent_cnn_feature(graph, snode_id)   # [句子数量, 128]

        # 得到 lstm_feature
        # features 是句向量列表, [[第1篇文章的37个句向量], [第2篇文章的30个句向量], ...(一共32即batch_size篇文章)],
        # glen 是句子数列表, [37, 30, 26, 14, 12, ...(32即batch_size个)]
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)   # [句子数量, 128]

        # 得到 node_feature (由 cnn_feature 和 lstm_feature 拼接而成)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2] [所有句子数量, 128*2]
        return node_feature


class HSumDocGraph(HSumGraph):
    """
        without sent2sent and add residual connection
        add Document Nodes
    """

    def __init__(self, hps, embed):
        super().__init__(hps, embed)
        self.dn_feature_proj = nn.Linear(hps.hidden_size, hps.hidden_size, bias=False)
        self.wh = nn.Linear(self.n_feature * 2, 2)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, type=0
                word2doc, doc2word: tffrac=int, type=0
                sent2doc: type=2
        :return: result: [sentnum, 2]
        """

        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        supernode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [snode, n_feature_size]

        # sent and doc node init
        graph.nodes[snode_id].data["init_feature"] = sent_feature
        doc_feature, snid2dnid = self.set_dnfeature(graph)
        doc_feature = self.dn_feature_proj(doc_feature)
        graph.nodes[dnode_id].data["init_feature"] = doc_feature

        # the start state
        word_state = word_feature
        sent_state = graph.nodes[supernode_id].data["init_feature"]
        sent_state = self.word2sent(graph, word_state, sent_state)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        graph.nodes[supernode_id].data["hidden_state"] = sent_state

        # extract sentence nodes
        s_state_list = []
        for snid in snode_id:
            d_state = graph.nodes[snid2dnid[int(snid)]].data["hidden_state"]
            s_state = graph.nodes[snid].data["hidden_state"]
            s_state = torch.cat([s_state, d_state], dim=-1)
            s_state_list.append(s_state)

        s_state = torch.cat(s_state_list, dim=0)
        result = self.wh(s_state)
        return result

    def set_dnfeature(self, graph):
        """ init doc node by mean pooling on the its sent node (connected by the edges with type=1) """
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        node_feature_list = []
        snid2dnid = {}
        for dnode in dnode_id:
            snodes = [nid for nid in graph.predecessors(dnode) if graph.nodes[nid].data["dtype"]==1]
            doc_feature = graph.nodes[snodes].data["init_feature"].mean(dim=0)
            assert not torch.any(torch.isnan(doc_feature)), "doc_feature_element"
            node_feature_list.append(doc_feature)
            for s in snodes:
                snid2dnid[int(s)] = dnode
        node_feature = torch.stack(node_feature_list)
        return node_feature, snid2dnid


def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)   # 得到这一篇文章的所有句子节点的 id
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen
