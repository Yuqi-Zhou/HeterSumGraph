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

import argparse
import datetime
import os
import time
import json

import torch
import torch.nn as nn
from rouge import Rouge

from HiGraph import HSumGraph, HSumDocGraph
from Tester import SLTester
from module.dataloader import ExampleSet, MultiExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools import utils
from tools.logger import *
from platform import platform   # ------------------------------------ 我自己加的
import jsonlines   # ------------------------------------ 我自己加的
my_num_workers_my = 0   # ------------------------------------ 我自己加的
from my_else import print_variable_info, my_dividing_line, my_stop   # ------------------------------------ 我自己加的


def load_test_model(model, model_name, eval_dir, save_root):
    """ choose which model will be loaded for evaluation """
    describe_str = './evaluation.py, 函数 load_test_model: '  # ------------------------------------ 我自己加的
    #  load_test_model(model, 'evalbestmodel_0', './dd_save/eval', './dd_save'), 其中参数model_name其实就是传入参数 --test_model
    if model_name.startswith('eval'):   # 如果传入参数 --test_model 以 eval 开头 (包括 --test_model 为 multi ), 比如为 evalbestmodel_0
        bestmodel_load_path = os.path.join(eval_dir, model_name[4:])    # bestmodel_load_path 为 './dd_save/eval/bestmodel_0'
    elif model_name.startswith('train'):    # 如果传入参数 --test_model 以 train 开头, 比如为 trainbestmodel
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, model_name[5:])   # bestmodel_load_path 为 './dd_save/train/bestmodel'
    elif model_name == "earlystop":  # 如果传入参数 --test_model 为 earlystop
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop')  # bestmodel_load_path 为 './dd_save/train/earlystop'
    else:
        logger.error(describe_str + "None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError(describe_str + "None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    if not os.path.exists(bestmodel_load_path):
        logger.error(describe_str + "[ERROR] Restoring %s for testing...The path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info(describe_str + "[INFO] Restoring %s for testing...The path is %s", model_name, bestmodel_load_path)

    model.load_state_dict(torch.load(bestmodel_load_path))

    return model


def run_test(model, dataset, loader, model_name, hps):
    describe_str = './evaluation.py, 函数 run_test: '  # ------------------------------------ 我自己加的
    test_dir = os.path.join(hps.save_root, "test")  # 测试模型目录, 为 ./dd_save/test
    eval_dir = os.path.join(hps.save_root, "eval")  # 评估模型目录, 为 ./dd_save/eval
    if not os.path.exists(test_dir) : os.makedirs(test_dir)
    if not os.path.exists(eval_dir) :
        logger.exception(describe_str + "[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception(describe_str + "[Error] eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    resfile = None
    if hps.save_label:
        log_dir = os.path.join(test_dir, hps.cache_dir.split("/")[-1])
        resfile = open(log_dir, "w")
        logger.info(describe_str + "[INFO] Write the Evaluation into %s", log_dir)

    #  model = load_test_model(model, 'evalbestmodel_0', './dd_save/eval', './dd_save')
    model = load_test_model(model, model_name, eval_dir, hps.save_root)
    model.eval()

    iter_start_time=time.time()
    # 使用 model 得到 tester
    with torch.no_grad():
        logger.info(describe_str + "[Model] Sequence Labeling!")
        tester = SLTester(model, hps.m, limited=hps.limited, test_dir=test_dir)

        for i, (G, index) in enumerate(loader):  # 这里的 loader 是测试集的 loader, 所以返回的是测试集的图G和index
            if hps.cuda:
                G.to(torch.device("cuda"))
            tester.evaluation(G, index, dataset, blocking=hps.blocking)

    running_avg_loss = tester.running_avg_loss

    if hps.save_label:
        # save label and do not calculate rouge 保存标签, 不计算 rouge
        json.dump(tester.extractLabel, resfile)
        tester.SaveDecodeFile()
        logger.info(describe_str + '   | end of test | time: {:5.2f}s | '.format((time.time() - iter_start_time)))
        return

    logger.info(describe_str + "The number of pairs is %d", tester.rougePairNum)
    if not tester.rougePairNum:
        logger.error(describe_str + "During testing, no hyps is selected!")
        sys.exit(1)

    # tester.hyps 和 tester.refer 都是列表, 列表有5000个元素(对应测试集的5000篇文摘, hyps是模型预测的文摘, refer是黄金文摘)
    # 每个元素都是一个字符串(表示文摘, 一个文摘的不同句子之间用\n分隔(linux), \r\n分隔(windows))
    if hps.use_pyrouge:
        if isinstance(tester.refer[0], list):
            logger.info(describe_str + "Multi Reference summaries!")
            # scores_all = utils.pyrouge_score_all_multi(tester.hyps, tester.refer)
            scores_all = utils.pyrouge_score_all_multi(tester.hyps, tester.refer, hps.pyrouge_temp_dir)    # ----------------------------------- 我自己修改上一行的
        else:
            # scores_all = utils.pyrouge_score_all(tester.hyps, tester.refer)
            scores_all = utils.pyrouge_score_all(tester.hyps, tester.refer, hps.pyrouge_temp_dir)    # ----------------------------------- 我自己修改上一行的
    else:
        # from rouge import Rouge, rouge = Rouge(), 下一行的 rouge 是文件 rouge.py 里的类 Rouge 的对象
        rouge = Rouge()
        scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
            + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
                + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(describe_str + '\n' + res)

    tester.getMetric()
    tester.SaveDecodeFile()  # 保存两种的所有摘要到同一个文件中
    logger.info(describe_str + '[INFO] End of test | time: {:5.2f}s | test loss {:5.4f} | '.format((time.time() - iter_start_time),float(running_avg_loss)))


# #################################################↓↓↓↓↓↓    # ----------------------------------- 下面是我自己加的
def test_lead_m(test_file_path, save_dir, pyrouge_temp_dir, use_tri_blocking, m):
    """
    测试 lead
    :param test_file_path: 要测试 lead 的文件, 文件格式为 HSG 模型的测试集文件的格式(文件中也可无 label 键)
    :param save_dir: 最终存放的摘要文件的文件夹, 会在该文件夹中新建 test 文件夹, 以存放摘要
    :param pyrouge_temp_dir: 使用 pyrouge 进行打分, 打分过程中的临时文件夹
    :param use_tri_blocking: 是否使用 tri_blocking 策略
    :param m: 一般测试 lead3, 即 m==3
    :return: 无
    """
    describe_str = './evaluation.py, 函数 test_lead3: '

    def check_parameters():
        if not (os.path.exists(test_file_path) and os.path.isfile(test_file_path)):
            logger.info(describe_str + "%s 不是文件!" % test_file_path)
            exit(0)
        if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
            logger.info(describe_str + "%s 不是文件夹!" % save_dir)
            exit(0)
        if not (os.path.exists(pyrouge_temp_dir) and os.path.isdir(pyrouge_temp_dir)):
            logger.info(describe_str + "%s 不是文件夹!" % pyrouge_temp_dir)
            exit(0)

    def save_decode_file(test_dir, hyps_list, refer_list):
        import datetime
        now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 现在
        result_file_path = os.path.join(test_dir, now_time)
        with open(result_file_path, "wb") as resfile:
            for i in range(len(hyps_list)):
                resfile.write(b"[Reference]\t")
                resfile.write(refer_list[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"[Hypothesis]\t")
                resfile.write(hyps_list[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"\n")
                resfile.write(b"\n")

    def get_lead_hyps_str(text_list, de, m):
        def _get_ngrams(n, text_list):
            ngram_set = set()
            text_length = len(text_list)
            for i in range(text_length - n + 1):
                ngram_set.add(tuple(text_list[i:i + n]))
            return ngram_set

        def _block_tri(a_sent, hyps_list):
            tri_a_sent = _get_ngrams(3, a_sent.split())
            for a_hyps_str in hyps_list:
                tri_a_hyps_str = _get_ngrams(3, a_hyps_str.split())
                if len(tri_a_sent.intersection(tri_a_hyps_str)) > 0:
                    return True
            return False
        if not use_tri_blocking:
            return de.join(text_list[:m])
        else:   # 如果文章正文的一个句子不与 temp_list 中的任一个句子有三元组重合, 才要这个句子, 最多要 m 句
            temp_list = []
            for a_sent in text_list:
                if not _block_tri(a_sent, temp_list):
                    temp_list.append(a_sent)
                if len(temp_list) >= m:
                    break
            return de.join(temp_list)

    check_parameters()
    test_dir = os.path.join(save_dir, 'test')
    if not (os.path.exists(test_dir) and os.path.isdir(test_dir)):
        os.makedirs(test_dir)

    test_data_list = [line for line in jsonlines.open(test_file_path)]

    # hyps_list 是模型预测的文摘(对于 lead, 即取前m句), refer_list 是黄金文摘
    hyps_list = []
    refer_list = []
    if 'win' in str(platform()).lower():   # de 是一个摘要字符串中不同句子之间的分隔符
        de = '\r\n'
    else:
        de = '\n'
    for example_dict in test_data_list:
        summary_list = example_dict["summary"]
        summary_str = de.join(summary_list)
        refer_list.append(summary_str)
        text_list = example_dict["text"]
        hyps_str = get_lead_hyps_str(text_list, de, m)
        hyps_list.append(hyps_str)
    assert (len(hyps_list) == len(refer_list)) and hyps_list

    scores_all = utils.pyrouge_score_all(hyps_list, refer_list, pyrouge_temp_dir)
    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
            + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
                + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(describe_str + '\n' + res)

    save_decode_file(test_dir, hyps_list, refer_list)
# #################################################↑↑↑↑↑↑    # ----------------------------------- 上面是我自己加的


# #################################################↓↓↓↓↓↓    # ----------------------------------- 下面是我自己加的
def test_oracle(test_file_path, save_dir, pyrouge_temp_dir, use_tri_blocking):
    """
    测试 oracle
    :param test_file_path: 要测试 oracle 的文件, 文件格式为 HSG 模型的测试集文件的格式
    :param save_dir: 最终存放的摘要文件的文件夹, 会在该文件夹中新建 test 文件夹, 以存放摘要
    :param pyrouge_temp_dir: 使用 pyrouge 进行打分, 打分过程中的临时文件夹
    :param use_tri_blocking: 是否使用 tri_blocking 策略
    :return: 无
    """
    describe_str = './evaluation.py, 函数 test_oracle: '

    def check_parameters():
        if not (os.path.exists(test_file_path) and os.path.isfile(test_file_path)):
            logger.info(describe_str + "%s 不是文件!" % test_file_path)
            exit(0)
        if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
            logger.info(describe_str + "%s 不是文件夹!" % save_dir)
            exit(0)
        if not (os.path.exists(pyrouge_temp_dir) and os.path.isdir(pyrouge_temp_dir)):
            logger.info(describe_str + "%s 不是文件夹!" % pyrouge_temp_dir)
            exit(0)

    def save_decode_file(test_dir, hyps_list, refer_list):
        import datetime
        now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 现在
        result_file_path = os.path.join(test_dir, now_time)
        with open(result_file_path, "wb") as resfile:
            for i in range(len(hyps_list)):
                resfile.write(b"[Reference]\t")
                resfile.write(refer_list[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"[Hypothesis]\t")
                resfile.write(hyps_list[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"\n")
                resfile.write(b"\n")

    def get_oracle_hyps_str(text_list, label_list, de):
        def _get_ngrams(n, text_list):
            ngram_set = set()
            text_length = len(text_list)
            for i in range(text_length - n + 1):
                ngram_set.add(tuple(text_list[i:i + n]))
            return ngram_set

        def _block_tri(a_sent, hyps_list):
            tri_a_sent = _get_ngrams(3, a_sent.split())
            for a_hyps_str in hyps_list:
                tri_a_hyps_str = _get_ngrams(3, a_hyps_str.split())
                if len(tri_a_sent.intersection(tri_a_hyps_str)) > 0:
                    return True
            return False

        temp_list = []
        for a_label in label_list:
            if use_tri_blocking and _block_tri(text_list[a_label], temp_list):
                continue
            temp_list.append(text_list[a_label])
        return de.join(temp_list)

    check_parameters()
    test_dir = os.path.join(save_dir, 'test')
    if not (os.path.exists(test_dir) and os.path.isdir(test_dir)):
        os.makedirs(test_dir)

    test_data_list = [line for line in jsonlines.open(test_file_path)]

    # hyps_list 是模型预测的文摘(对于 oracle, 即取标签所指的句子), refer_list 是黄金文摘
    hyps_list = []
    refer_list = []
    if 'win' in str(platform()).lower():    # de 是一个摘要字符串中不同句子之间的分隔符
        de = '\r\n'
    else:
        de = '\n'
    for example_dict in test_data_list:
        summary_list = example_dict["summary"]
        summary_str = de.join(summary_list)
        refer_list.append(summary_str)

        text_list = example_dict["text"]
        label_list = example_dict["label"]
        hyps_str = get_oracle_hyps_str(text_list, label_list, de)
        hyps_list.append(hyps_str)
    assert (len(hyps_list) == len(refer_list)) and hyps_list

    scores_all = utils.pyrouge_score_all(hyps_list, refer_list, pyrouge_temp_dir)
    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
            + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
                + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(describe_str + '\n' + res)

    save_decode_file(test_dir, hyps_list, refer_list)
# #################################################↑↑↑↑↑↑    # ----------------------------------- 上面是我自己加的


class ARGS:  # ------------------------------------ 我自己加的, 用于方便地进行参数调整
    def __init__(self):
        self.atten_dropout_prob = 0.1
        self.batch_size = 32
        self.bidirectional = True

        # self.blocking = False
        self.blocking = True

        self.cache_dir = './aa_data_graphfile/cache/试验'
        self.cuda = False
        self.data_dir = './aa_data_datasets/试验'
        self.doc_max_timesteps = 50
        self.embed_train = False
        self.embedding_path = './aa_word_vector_file/chinese_merge_SGNS/(小)word+chr+Ngram____merge_sgns_bigram_char300.txt'
        self.feat_embed_size = 50
        self.ffn_dropout_prob = 0.1
        self.ffn_inner_hidden_size = 512
        self.gcn_hidden_size = 128
        self.gpu = '0'
        self.hidden_size = 64
        self.interest_dir = './aa_interest/试验'
        self.language = 'chinese'
        self.limited = False
        self.log_root = './oracle+cles(words)/bb_log'
        self.lstm_hidden_state = 128
        self.lstm_layers = 2
        self.m = 3
        self.model = 'oracle'
        self.n_feature_size = 128
        self.n_head = 8
        self.n_iter = 1
        self.n_layers = 1
        self.pyrouge_temp_dir = './oracle+cles(words)/bb_pyrouge_temp'
        self.recurrent_dropout_prob = 0.1
        self.save_label = False
        self.save_root = './oracle+cles(words)/bb_save'
        self.sent_max_len = 100

        self.test_model = 'trainbestmodel'
        # self.test_model = 'evalbestmodel'

        self.use_interest = True
        # self.use_interest = False

        self.use_orthnormal_init = True

        self.use_pyrouge = True
        # self.use_pyrouge = False

        self.vocab_size = 50000
        self.word_emb_dim = 300
        self.word_embedding = True


def main():
    describe_str = './evaluation.py, 函数 main: '  # ------------------------------------ 我自己加的
    if 'win' in str(platform()).lower():       # ------------------------------------ 我自己加的(windows)
        args = ARGS()  # ------------------------------------ 我自己加的
    else:  # ------------------------------------ 我自己加的(linux)
        parser = argparse.ArgumentParser(description='HeterSumGraph Model')

        # Where to find data
        parser.add_argument('--data_dir', type=str, default='data/CNNDM', help='The dataset directory.')
        parser.add_argument('--cache_dir', type=str, default='cache/CNNDM', help='The processed dataset directory')
        parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

        # #################################################↓↓↓↓↓↓    # ----------------------------------- 下面是我自己加的
        # 我自己加的几个参数
        parser.add_argument('--use_interest', action='store_true', default=False, help='是否使用兴趣图谱, 默认为否')
        parser.add_argument('--interest_dir', type=str, default='./aa_interest/simplified_cles_interest', help='三个简化后的兴趣图谱文件所在的文件夹路径')
        parser.add_argument('--pyrouge_temp_dir', type=str, default='./pyrouge_temp_dir', help='pyrouge (非rouge) 算分数的临时文件夹')
        parser.add_argument('--language', type=str, default='chinese', help='仅用于在 dataloader.py 里判断加载停用词的时候加载哪种语言的停用词. 仅值为 chinese 时加载中文停用词, 其他的都加载英文停用词')
        # #################################################↑↑↑↑↑↑    # ----------------------------------- 上面是我自己加的

        # Important settings
        parser.add_argument('--model', type=str, default="HSG", help="model structure[HSG|HDSG](或lead, 或oracle)")  # 原先的 default 写成 HSumGraph, 现改为 HSG
        parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [multi/evalbestmodel/trainbestmodel/earlystop]')
        parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')

        # Where to save output
        parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
        parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

        # Hyperparameters
        parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
        parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
        parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary.')
        parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
        parser.add_argument('--n_iter', type=int, default=1, help='iteration ')

        parser.add_argument('--word_embedding', action='store_true', default=True, help='whether to use Word embedding')
        parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
        parser.add_argument('--embed_train', action='store_true', default=False, help='whether to train Word embedding [default: False]')
        parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
        parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
        parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state')
        parser.add_argument('--lstm_layers', type=int, default=2, help='lstm layers')
        parser.add_argument('--bidirectional', action='store_true', default=True, help='use bidirectional LSTM')
        parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature')
        parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
        parser.add_argument('--gcn_hidden_size', type=int, default=128, help='hidden size [default: 64]')
        parser.add_argument('--ffn_inner_hidden_size', type=int, default=512, help='PositionwiseFeedForward inner hidden size [default: 512]')
        parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
        parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1, help='recurrent dropout prob [default: 0.1]')
        parser.add_argument('--atten_dropout_prob', type=float, default=0.1,help='attention dropout prob [default: 0.1]')
        parser.add_argument('--ffn_dropout_prob', type=float, default=0.1, help='PositionwiseFeedForward dropout prob [default: 0.1]')
        parser.add_argument('--use_orthnormal_init', action='store_true', default=True, help='use orthnormal init for lstm [default: true]')
        parser.add_argument('--sent_max_len', type=int, default=100, help='max length of sentences (max source text sentence tokens)')
        parser.add_argument('--doc_max_timesteps', type=int, default=50, help='max length of documents (max timesteps of documents)')
        parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')
        parser.add_argument('--limited', action='store_true', default=False, help='limited hypo length')
        parser.add_argument('--blocking', action='store_true', default=False, help='ngram blocking')

        parser.add_argument('-m', type=int, default=3, help='decode summary length')

        args = parser.parse_args()

    assert args.model in ["HSG", "HDSG", 'lead', 'oracle']  # ------------------------------------ 我自己加的
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu   # 不设置则默认为0
    torch.set_printoptions(threshold=50000)  # 设置输出格式, threshold=50000 意思是当tensor中元素的个数大于5000时，进行缩略输出

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "test.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")
    LOG_PATH = args.log_root

    # 设置 log 文件, log 根文件夹必须存在
    if not os.path.exists(LOG_PATH):
        # logger.exception(describe_str + "[Error] Logdir %s doesn't exist. Run in train mode to create it.", LOG_PATH)
        logger.exception(describe_str + "[Error] --log_root 指定的 log 目录 %s 不存在, 退出程序", args.log_root)  # ------ 我自己修改上一行的
        # raise Exception(describe_str + "[Error] Logdir %s doesn't exist. Run in train mode to create it" % (LOG_PATH))
        raise Exception(describe_str + "[Error] --log_root 指定的 log 目录 %s 不存在, 退出程序", args.log_root)  # ------- 我自己修改上一行的
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # log_path = os.path.join(LOG_PATH, "test_" + nowTime)    # ----------------------- 将本行修改为下面几行
    if args.model in ["HSG", "HDSG"]:
        log_path = os.path.join(LOG_PATH, "test_" + nowTime + '(测试' + args.test_model + ')')
    elif args.model == "lead":
        log_path = os.path.join(LOG_PATH, "test_lead_" + nowTime)
    elif args.model == 'oracle':
        log_path = os.path.join(LOG_PATH, "test_oracle_" + nowTime)
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')  # ------------------- 多加了 mode 和 enconding
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # #################################################↓↓↓↓↓↓    # ----------------------------------- 下面是我自己加的
    logger.info(describe_str + 'model: ' + str(args.model))
    if args.model in ["HSG", "HDSG"]:
        logger.info(describe_str + 'test_model: ' + str(args.test_model))
        logger.info(describe_str + 'use_interest: ' + str(args.use_interest))
    logger.info(describe_str + 'use_pyrouge: ' + str(args.use_pyrouge))
    logger.info(describe_str + 'Tri-Blocking: ' + str(args.blocking))

    if args.model == 'lead':   # 使用 lead3 测试
        if not args.use_pyrouge:
            logger.info(describe_str + "当模型为 lead 时, 必须指定 --use_pyrouge (指定即为True). 没有改动任何文件, 现退出程序")
            exit(0)
        if not args.m == 3:
            logger.info(describe_str + "当模型为 lead 时, 必须指定 -m 为 3. 没有改动任何文件, 现退出程序")
            exit(0)
        test_lead_m(DATA_FILE, args.save_root, args.pyrouge_temp_dir, args.blocking, args.m)
        exit(0)
    elif args.model == 'oracle':    # 使用 oracle 测试
        if not args.use_pyrouge:
            logger.info(describe_str + "当模型为 oracle 时, 必须指定 --use_pyrouge (指定即为True). 没有改动任何文件, 现退出程序")
            exit(0)
        test_oracle(DATA_FILE, args.save_root, args.pyrouge_temp_dir, args.blocking)
        exit(0)
    # #################################################↑↑↑↑↑↑    # ----------------------------------- 上面是我自己加的

    logger.info(describe_str + "Pytorch %s", torch.__version__)
    logger.info(describe_str + "[INFO] 创建词典, 词典路径是 %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim)
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train

    # 获取 hps
    hps = args
    logger.info(describe_str + '参数设置:' + str(hps))

    test_w2s_path = os.path.join(args.cache_dir, "test.w2s.tfidf.jsonl")
    test_interest_path = os.path.join(args.interest_dir, 'test_interest.jsonl')  # ----------------------------- 我自己加的
    if hps.model == "HSG":
        model = HSumGraph(hps, embed)
        logger.info(describe_str + "[模型] HeterSumGraph: \n" + str(model))
        # dataset = ExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path)
        dataset = ExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path,
                             args.use_interest, test_interest_path, hps.language)  # ------------------------- 我自己修改上一行的
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=my_num_workers_my,collate_fn=graph_collate_fn)
    elif hps.model == "HDSG":  # 多文档的没有加兴趣图谱
        model = HSumDocGraph(hps, embed)
        logger.info(describe_str + "[MODEL] HeterDocSumGraph ")
        test_w2d_path = os.path.join(args.cache_dir, "test.w2d.tfidf.jsonl")
        dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path, test_w2d_path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=my_num_workers_my,collate_fn=graph_collate_fn)
    else:
        logger.error(describe_str + "[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    if args.cuda:
        model.to(torch.device("cuda:0"))
        logger.info(describe_str + "[INFO] Use cuda")

    logger.info(describe_str + "[INFO] Decoding...")
    if hps.test_model == "multi":
        for i in range(3):
            model_name = "evalbestmodel_%d" % i
            run_test(model, dataset, loader, model_name, hps)
    else:
        run_test(model, dataset, loader, hps.test_model, hps)


if __name__ == '__main__':
    main()
