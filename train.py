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
import shutil
import time

import dgl
import numpy as np
import torch
from rouge import Rouge

from HiGraph import HSumGraph, HSumDocGraph
from Tester import SLTester
from module.dataloader import ExampleSet, MultiExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *
from my_else import MyLogger  # ------------------------------------ 我自己加的

_DEBUG_FLAG_ = False

import sys

def save_model(model, save_file):
    describe_str = './train.py, 函数 save_model: '  # ------------------------------------ 我自己加的
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info(describe_str + '[INFO] Saving model to %s', save_file)


def setup_training(model, train_loader, valid_loader, valset, hps):
    """ Does setup before starting training (run_training)
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :return: 
    """
    describe_str = './train.py, 函数 setup_training: '
    train_dir = os.path.join(hps.save_root, "train")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info(describe_str + "[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir,
                                      hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        logger.info(describe_str + "[INFO] Create new model for training...")
        if os.path.exists(train_dir): shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, train_loader, valid_loader, valset, hps, train_dir)
    except KeyboardInterrupt:
        logger.error(describe_str + "[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


def run_training(model, train_loader, valid_loader, valset, hps, train_dir):
    '''  Repeatedly runs training iterations, logging loss to screen and log files
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param train_dir: where to save checkpoints
        :return: 
    '''
    describe_str = './train.py, 函数 run_training: '  # ------------------------------------ 我自己加的
    logger.info(describe_str + "[INFO] Starting run_training")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0

    for epoch in range(1, hps.n_epochs + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        epoch_start_time = time.time()
        for i, (G, index) in enumerate(train_loader):
            # 这里的 i 是0, 随着循环的进行会变成1, 2, 3, ...
            # 这里的 G 是32(batch_size)个G拼接起来的, 拼接顺序就是按 index 中的前后顺序拼接的
            # 这里的 index 是文章索引列表, 如 [34, 78, 21, 8, ..., 22, 56, 一共有32(batch_size)个整数]
            # if not index == [0]:    # 配合batch_size为1, 强制锁定训练集的第一篇文章(大公司头条的微信公众号上线啦)    # ---------------- 我自己加的
            #     continue    # ----------------------------------- 我自己加的
            iter_start_time = time.time()
            model.train()  # 类 nn.Module 的 train 方法

            if hps.cuda:
                G.to(torch.device("cuda"))

            # 前向传播求出预测文摘
            outputs = model.forward(G)  # [句子数量, 2]

            snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)  # 得到所有的句子节点 id
            label = G.ndata["label"][snode_id].sum(-1)  # 得到一个标签列表 [0, 0, 1, 0, ...(一共有句子数个元素, 为1表示该句是文摘)]

            # 求 loss(将outputs, label进行比对)
            G.nodes[snode_id].data["loss"] = criterion(outputs, label).unsqueeze(-1)  # 给所有的句子节点加特征loss, 值为 [浮点数(0~2)]

            # [batch_size, 1]  # 将每篇文章的 loss 属性分别相加, 得到 [文章0的loss, 文章1的loss... (一共batch_size篇文章)]
            loss = dgl.sum_nodes(G, "loss")
            loss = loss.mean()  # 对 loss 列表进行平均(相加再除以 batch_size), 得到一个 浮点数 loss

            if not (np.isfinite(loss.data.cpu())).numpy():
                logger.error(describe_str + "训练 Loss 无穷, 停止训练. train Loss is not finite. Stopping.")
                logger.info(describe_str + 'loss: ' + str(loss))
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(describe_str + 'name: ' + str(name))
                        # logger.info(param.grad.data.sum())
                raise Exception(describe_str + "训练 Loss 无穷, 停止训练. train Loss is not finite. Stopping.")

            optimizer.zero_grad()  # 梯度初始化为 0
            loss.backward()  # 反向传播求梯度
            if hps.grad_clip:  # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)

            optimizer.step()  # 更新所有参数

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i % 100 == 0:  # 每 100个 iter(iter即batch, 一个batch32篇文章) 打印一次, 一轮下来打印 29 次
                if _DEBUG_FLAG_:  # 全局变量, 设为 False
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.debug(describe_str + 'name: ' + str(name))
                            logger.debug(describe_str + 'param.grad.data.sum(): ' + str(param.grad.data.sum()))
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                            .format(i, (time.time() - iter_start_time), float(train_loss / 100)))
                train_loss = 0.0

        logger.info('1 个 epoch 结束, 下面是总结')  # ------------------------------------ 我自己加的
        if hps.lr_descent:  # 默认为假, 指定为真
            new_lr = max(5e-6, hps.lr / (epoch + 1))  # 更新学习率, 最初的学习率默认为 0.0005 (学习率逐渐降低), 可以在 epoch 前加倍数来控制学习率变化
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info(describe_str + "[INFO] The learning rate now is %f", new_lr)

        epoch_avg_loss = epoch_loss / len(train_loader)  # 平均到每个 batch 的loss
        logger.info(describe_str + '\n' + '   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        # 训练集 loss 下降, 保存 bestmodel, 继续训练; 训练集 loss 不下降保存 earlystop, 并结束程序
        if not best_train_loss or epoch_avg_loss < best_train_loss:
            nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # ------------------------------------ 我自己加的
            # save_file = os.path.join(train_dir, "bestmodel")
            save_file = os.path.join(train_dir, "bestmodel" + '_' + nowTime)    # ---------------------------- 我自己修改上一行的
            logger.info(describe_str + '[INFO] Found new best model with %.3f running_train_loss. Saving to %s',
                        float(epoch_avg_loss),
                        save_file)
            save_model(model, save_file)
            best_train_loss = epoch_avg_loss
        elif epoch_avg_loss >= best_train_loss:
            logger.error(describe_str + "[Error] training loss does not descent. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"))
            # if send_mail_when_finished:
            # send_mail('epoch_avg_loss >= best_train_loss, 训练集 loss 不下降, 退出程序')  # -------------------------------- 我自己加的
            sys.exit(1)

        # 在验证集上测 rouge 分数
        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valset, hps, best_loss, best_F,
                                                              non_descent_cnt, saveNo)

        if non_descent_cnt <= 2:  # -------------------------------- 我自己加的
            logger.info(describe_str + '\n验证集上连续 %d 次 loss\n' % non_descent_cnt)  # -------------------------------- 我自己加的
        else:  # -------------------------------- 我自己加的
            da_info = describe_str + '\n验证集上连续 %d 次 loss\n' % non_descent_cnt  # -------------------------------- 我自己加的
            logger.info(da_info)  # -------------------------------- 我自己加的
            # send_mail(da_info)  # ------------------------------------ 我自己加的
        # if non_descent_cnt >= 3:
        if non_descent_cnt >= 6:  # 我自己修改上一行的.
            # logger.error(describe_str + "[Error] val loss does not descent for three times. Stopping supervisor...")
            logger.error(describe_str + "验证集上连续6次 loss 不下降, 保存模型 earlystop , 退出程序")  # ----------------- 我自己修改上一行的
            save_model(model, os.path.join(train_dir, "earlystop"))
            return


def run_eval(model, loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo):
    ''' 
        Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.
        :param model: the model
        :param loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param best_loss: best valid loss so far
        :param best_F: best valid F so far
        :param non_descent_cnt: the number of non descent epoch (for early stop)
        :param saveNo: the number of saved models (always keep best saveNo checkpoints)
        :return: 
    '''
    describe_str = './train.py, 函数 run_eval: '  # ------------------------------------ 我自己加的
    logger.info(describe_str + "[INFO] Starting eval for this model ...")
    eval_dir = os.path.join(hps.save_root, "eval")  # make a subdir of the root dir for eval data
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    model.eval()

    iter_start_time = time.time()

    with torch.no_grad():
        tester = SLTester(model, hps.m)
        for i, (G, index) in enumerate(loader):  # 这个 loader 是验证集的 loader, 所以会把验证集的 图G和index 返回回来
            if hps.cuda:
                G.to(torch.device("cuda"))
            tester.evaluation(G, index, valset)     # 验证的时候, 不使用 Tri-Blocking 策略. 如果要使用, 请在本行参数中设置 blocking = True

    running_avg_loss = tester.running_avg_loss

    if len(tester.hyps) == 0 or len(tester.refer) == 0:
        logger.error(describe_str + "During testing, no hyps is selected!")
        return

    # rouge = Rouge()
    # scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)
    from tools import utils  # ---------------- 我自己修改上面两行的
    scores_all = utils.pyrouge_score_all(tester.hyps, tester.refer, hps.pyrouge_temp_dir)  # ---------------- 我自己修改上面两行的

    logger.info(describe_str + '\n' + '[INFO] End of valid | time: {:5.2f}s | valid loss {:5.4f} | '.format(
        (time.time() - iter_start_time), float(running_avg_loss)))

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    tester.getMetric()
    F = tester.labelMetric

    if best_loss is None or running_avg_loss < best_loss:
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # ------------------------------------ 我自己加的
        # bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % (saveNo % 3))
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % (saveNo % 3) + '_' + nowTime)  # ------- 我自己修改上一行的
        if best_loss is not None:
            # logger.info(describe_str + '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
            #     float(running_avg_loss), float(best_loss), bestmodel_save_path)
            logger.info(describe_str + '[INFO] 找到了新的最好模型, running_avg_loss 为: %.6f. (比原先的 best_loss %.6f 好). 将模型保存到 %s',
                        float(running_avg_loss), float(best_loss), bestmodel_save_path)

        else:
            # logger.info(describe_str + '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
            #     float(running_avg_loss), bestmodel_save_path)
            logger.info(describe_str + '[INFO] 找到了新的最好模型, running_avg_loss 为: %.6f. (原先的 best_loss 是 None). 将模型保存到 %s',
                        float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        logger.info(describe_str + '将 best_loss 从 %s 更新为 %s (best_loss = running_avg_loss)' % (
        str(best_loss), str(running_avg_loss)))  # ------------------------------------ 我自己加的
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    else:
        non_descent_cnt += 1

    if best_F is None or best_F < F:
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # ------------------------------------ 我自己加的
        # bestmodel_save_path = os.path.join(eval_dir, 'bestFmodel')
        bestmodel_save_path = os.path.join(eval_dir, 'bestFmodel' + '_' + nowTime)  # -------------- 我自己修改上一行的
        if best_F is not None:
            # logger.info(describe_str + '[INFO] Found new best model with %.6f F. The original F is %.6f, Saving to %s', float(F),
            #             float(best_F), bestmodel_save_path)
            logger.info(describe_str + '[INFO] 找到了新的最好模型, F 为 %.6f. (比原先的 best_F %.6f 好). 将模型保存到 %s', float(F),
                        float(best_F), bestmodel_save_path)
        else:
            logger.info(describe_str + '[INFO] 找到了新的最好模型, F 为 %.6f .(比原先的 best_F None 好). 将模型保存到 %s', float(F),
                        bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        logger.info(describe_str + '将 best_F 从 %s 更新为 %s (best_F = F)' % (
        str(best_F), str(F)))  # ------------------------------------ 我自己加的
        best_F = F

    return best_loss, best_F, non_descent_cnt, saveNo

def main():
    describe_str = './train.py, 函数 main: '  # ------------------------------------ 我自己加的
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='./aa_data_datasets/cles_word_chinese',help='数据集路径The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='./aa_data_graphfile/cache/CLES_word_chinese',help='图数据路径The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='./aa_word_vector_file/chinese_merge_SGNS/word+chr+Ngram____merge_sgns_bigram_char300.txt', help='外部词向量路径 Path expression to external word embedding.')

    # 我自己加的几个参数
    parser.add_argument('--use_interest', action='store_true', default=False, help='是否使用兴趣图谱, 默认为否')    # ----------------------------------- 我自己加的
    parser.add_argument('--interest_dir', type=str, default='./aa_interest/simplified_cles_interest', help='三个简化后的兴趣图谱文件所在的文件夹路径')    # ----------------------------------- 我自己加的
    parser.add_argument('--pyrouge_temp_dir', type=str, default='./HSG+cles+ig/bb_pyrouge_temp', help='pyrouge (非rouge) 算分数的临时文件夹')    # ----------------------------------- 我自己加的
    parser.add_argument('--language', type=str, default='chinese', help='仅用于在 dataloader.py 里判断加载停用词的时候加载哪种语言的停用词. 仅值为 chinese 时加载中文停用词, 其他的都加载英文停用词')  # ----------------------------------- 我自己加的

    # Important settings
    parser.add_argument('--model', type=str, default='HSG', help='模型结构 model structure[HSG|HDSG]')
    parser.add_argument('--restore_model', type=str, default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='./HSG+cles+ig/bb_save', help='用来保存数据的根目录 Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='./HSG+cles+ig/bb_log', help='用来保存日志的根目录 Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=True, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=50000, help='从词典中读取的词数(行数) Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='默认跑20轮 Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    parser.add_argument('--word_embedding', action='store_true', default=True, help='是否使用外部词向量 whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='一个词的词向量长度 Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False, help='是否训练词向量 whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='GAT 层的数量 Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='lstm 层的数量 Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='whether to use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512, help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1, help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1, help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True, help='use orthnormal init for lstm [default: True]')
    parser.add_argument('--sent_max_len', type=int, default=100, help='一个句子的最大令牌数 max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50, help='一篇文档的最大句子数 max length of documents (max timesteps of documents)')

    # Training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='是否要学习率下降 learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='梯度裁剪 for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')

    parser.add_argument('-m', type=int, default=3, help='抽取的摘要的句数 decode summary length')

    args = parser.parse_args()
    # ---------- 变量名: args , 类型: <class 'argparse.Namespace'> , 值: Namespace(atten_dropout_prob=0.1, ...)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 不设置则默认为0
    torch.set_printoptions(threshold=50000)  # 设置输出格式, threshold=50000 意思是当tensor中元素的个数大于5000时，进行缩略输出

    # 得到文件路径
    DATA_FILE = os.path.join(args.data_dir, "train.label.jsonl")
    VALID_FILE = os.path.join(args.data_dir, "val.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")
    LOG_PATH = args.log_root
    # ---------- 变量名: DATA_FILE , 类型: <class 'str'> , 长度: 42 , 值: ./aa_data_datasets/xxxxxx/train.label.jsonl
    # ---------- 变量名: VALID_FILE , 类型: <class 'str'> , 长度: 40 , 值: ./aa_data_datasets/xxxxxx/val.label.jsonl
    # ---------- 变量名: VOCAL_FILE , 类型: <class 'str'> , 长度: 37 , 值: ./aa_data_graphfile/cache/xxxxxx/vocab
    # ---------- 变量名: FILTER_WORD , 类型: <class 'str'> , 长度: 47 , 值: ./aa_data_graphfile/cache/xxxxxx/filter_word.txt
    # ---------- 变量名: LOG_PATH , 类型: <class 'str'> , 长度: 8 , 值: ./xxxxxx/bb_log

    # 设置 log 文件
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    sys.stdout = MyLogger(os.path.join(LOG_PATH, "stdout_" + nowTime), sys.stdout, mode='w')  # ------------------ 我自己加的
    sys.stderr = MyLogger(os.path.join(LOG_PATH, "stderr_" + nowTime), sys.stderr, mode='w')  # ------------------ 我自己加的
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')  # ------------------- 多加了 mode 和 enconding
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 获取 vocab 词典对象 和 embed 词向量表对象
    logger.info(describe_str + "Pytorch %s", torch.__version__)
    logger.info(describe_str + "[INFO] 创建词典, 词典路径是 %s", VOCAL_FILE)
    # vocab 对象表示词典. vocab 对象将 ./aa_data_graphfile/cache/xxxxxx/vocab 里的内容变成两个字典 vocab._word_to_id 和 vocab._id_to_word
    # 其中{0: '[PAD]', 1: '[UNK]', 2: '[START]', 3: '[STOP]'}. 字典长度都为 args.vocab_size = 50000(实际词有 4996个, 4个被占据).
    # 举个例子, w='说味儿二欧文诶车次', 该词不在词典中, 则 vocab.word2id(w)为1(因为1代表未知), 而vocab.id2word(1)为'[UNK]', 就把'说味儿二欧文诶车次'变为了'[UNK]'
    # (如果 args.vocab_size = 50000 超过文件行数或者为0, 会读完整个vocab文件为止)
    # vocab = Vocab(./aa_data_graphfile/cache/xxxxxx/vocab, 50000)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    # embed 是随机生成的词向量表, embed.weight.data 就是 tensor 词向量表, 该表 50000 * 300
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim,
                               padding_idx=0)  # embed = torch.nn.Embedding(50000, 300, padding_idx=0)
    if args.word_embedding:  # 默认为真
        embed_loader = Word_Embedding(args.embedding_path,
                                      vocab)  # Word_Embedding('./aa_glove/glove.42B.300d.txt', vocab)
        # vectors 是字典 {'词':[词向量], ...}, 其中词是在vocab文件(前50000个词)和外部词向量文件中(不包括第一行)同时出现的词
        # 其中[词向量]的长度为 args.word_emb_dim = 300 (要把 args.word_emb_dim 设置成小于等于外部词典中词向量的最短长度)
        # 注: 如果外部词典够大, vectors 字典里的键就会基本上包含了vocab文件的前 50000个词汇
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)  # embed_loader.load_my_vecs(300)
        # pretrained_weight 是[[词向量1], [词向量2], ...], 前四个词向量是固定令牌词,
        # 从第5个词向量开始, 排序与 vocab 文件中的词一致且刚好有 50000 个词
        # vectors 一般不会恰巧 50000 个词都有, 故把没有的词的词向量用已有的词的词向量的平均值替代, 把 vectors 变成 pretrained_weight
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))  # 把 pretrained_weight 导入 embed 里
        embed.weight.requires_grad = args.embed_train  # args.embed_train 表示是否训练词向量, 默认 False, 不用再训练了
    logger.info(describe_str + 'embed 词向量表准备完成')  # ------------------------------------ 我自己加的
    # a = embed.weight.data[0]  # ------------------------------------ 我自己加的
    # print_variable_info(a)  # ------------------------------------ 我自己加的
    # b = embed.weight.data[4]  # 是逗号  # ------------------------------------ 我自己加的
    # print_variable_info(b)  # ------------------------------------ 我自己加的
    # my_stop()  # ------------------------------------ 我自己加的

    # 获取 hps
    hps = args
    logger.info(describe_str + '参数设置: ' + str(hps))

    train_w2s_path = os.path.join(args.cache_dir, "train.w2s.tfidf.jsonl")
    val_w2s_path = os.path.join(args.cache_dir, "val.w2s.tfidf.jsonl")
    # 变量名: train_w2s_path , 类型: <class 'str'> , 长度: 53 , 值: ./aa_data_graphfile/cache/xxxxxx/train.w2s.tfidf.jsonl
    # 变量名: val_w2s_path , 类型: <class 'str'> , 长度: 51 , 值: ./aa_data_graphfile/cache/xxxxxx/val.w2s.tfidf.jsonl
    train_interest_path = os.path.join(args.interest_dir, 'train_interest.jsonl')  # ----------------------------- 我自己加的
    val_interest_path = os.path.join(args.interest_dir, 'val_interest.jsonl')  # --------------------------------- 我自己加的

    # 根据 hps, vocab ,embed 获取 model, 两个 loader 和 一个 valid_dataset
    if hps.model == "HSG":
        # 建立模型 model 对象
        model = HSumGraph(hps, embed)
        logger.info(describe_str + '[模型] HeterSumGraph: \n' + str(model))
        # 加载训练集
        # dataset = ExampleSet( 等\train.label.jsonl, vocab, 50, 100,等\filter_word.txt, 等\train.w2s.tfidf.jsonl)
        # dataset = ExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path)
        dataset = ExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path,
                             args.use_interest, train_interest_path, hps.language)  # ----------------------------------- 我自己修改上一行的
        # 组合训练集和采样器, 并提供可迭代的返回值.
        # train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,collate_fn=graph_collate_fn(是个./module/dataloader的函数))
        # dataset(数据集): 要从中加载数据的数据集
        # batch_size(整数, 可选): 每个批次要加载多少个样本(默认1)
        # shuffle(布尔型, 可选): 设置为True以使数据在每个时期都重新随机播放(默认False)
        # num_workers(整数, 可选): 多少个子进程用于数据加载, 0表示将在主进程中加载​​数据
        # collat​​e_fn(可调用, 可选): 合并样本列表以形成张量的小批量. 在从地图样式数据集中使用批量加载时使用
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True,
                                                   num_workers=1, collate_fn=graph_collate_fn)
        del dataset
        # valid_dataset = ExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path)
        valid_dataset = ExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD,
                                   val_w2s_path, args.use_interest, val_interest_path, hps.language)  # ----------------------------------- 我自己修改上一行的
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False,
                                                   collate_fn=graph_collate_fn, num_workers=1)
    elif hps.model == "HDSG":  # 多文档的没有加兴趣图谱
        model = HSumDocGraph(hps, embed)
        logger.info(describe_str + "[模型] HeterDocSumGraph: " + str(model))
        train_w2d_path = os.path.join(args.cache_dir, "train.w2d.tfidf.jsonl")
        dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD,
                                  train_w2s_path, train_w2d_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True,
                                                   num_workers=1, collate_fn=graph_collate_fn)
        del dataset
        val_w2d_path = os.path.join(args.cache_dir, "val.w2d.tfidf.jsonl")
        valid_dataset = MultiExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD,
                                        val_w2s_path, val_w2d_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False,
                                                   collate_fn=graph_collate_fn,
                                                   num_workers=1)  # Shuffle Must be False for ROUGE evaluation
    else:
        logger.error(describe_str + "[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    if args.cuda:  # 默认为否
        model.to(torch.device("cuda:0"))
        logger.info(describe_str + "[INFO] Use cuda")

    # 传入参数
    setup_training(model, train_loader, valid_loader, valid_dataset, hps)


if __name__ == '__main__':
    main()