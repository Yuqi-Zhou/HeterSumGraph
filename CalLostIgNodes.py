import os
import time
import argparse

import json
import jsonlines

from module.vocabulary import Vocab

def check_dir(args):
    """

    """
    if not os.path.exists(args.ig_dir):
        print("The interest graph data directory doesn't exit.")
        exit(0)
    if not os.path.exists(args.data_dir):
        print("The data directory doesn't exit.")
        exit(0)
    # if not os.path.exists(args.simplified_ig_dir):
    #     os.makedirs(args.simplified_ig_dir)


def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def sent2word(sents):
    """

    """
    word_set = set()
    for sent in sents:
        word_set.update(sent.split(' '))
    words = list(word_set)
    return words

def cal_lost(igfile, datafile, sigfile, filterwords):

    print('igdata file: {}'.format(igfile))
    print('data file: {}'.format(datafile))

    begin_time = time.time()
    sig_articles = []
    with jsonlines.open(sigfile, 'r') as fs:
        for line in fs:
            sig_articles.append(line)

    with jsonlines.open(datafile, 'r') as fd:
        with open(igfile, 'r', encoding='utf-8') as fi:
            index = 0
            ig_articles = json.load(fi)
            all_ig_words_len = []
            ig_word_len = []
            for article in fd:
                texts = sent2word(article["text"])
                # get word nodes
                words = []
                for word in texts:
                    if (word not in filterwords) and (word not in words):
                        words.append(word)
                # get all id nodes
                # all_ig_words = []
                all_len = 0
                ig_article = ig_articles[index]
                word_nodes = ig_article['graph_info'][0]['nodes']
                for wnode in word_nodes:
                    if wnode['category'] != 'text':
                        # all_ig_words.append(wnode['text'])
                        all_len += 1
                all_ig_words_len.append(all_len)

                # get simplified ig words
                ig_words = []

                for i, type in enumerate(['s_text', 'd_text']):
                    for edge in sig_articles[index][type]:
                        word = edge[i]
                        ig_word = edge[1-i]
                        if (word in words) and (ig_word not in filterwords) and \
                            (ig_word not in ig_words) and (ig_word not in words):
                            ig_words.append(ig_word)
                ig_word_len.append(len(ig_words))
                index += 1

        assert len(all_ig_words_len) == len(ig_word_len)
        ignodes_sum = sum(all_ig_words_len)
        signdoes_sum = sum(ig_word_len)
        print("total igword's num is: {}".format(ignodes_sum))
        print("simplified igword's num is: {}".format(signdoes_sum))
        print("there are {} words that are be dropped.".format(ignodes_sum - signdoes_sum))
        print("the average dropped rate is {}".format(round((1 - signdoes_sum/ignodes_sum)*100)))




def main():
    parser = argparse.ArgumentParser(description='simplify_cles_interest_graph.py')

    parser.add_argument('--cache_dir', type=str, default='cache/CLES_word_chinese', help='The cache directory.')
    parser.add_argument('--data_dir', type=str, default='data/CLES_word_chinese', help='The dataset directory.')
    parser.add_argument('--ig_dir', type=str, default='interest/cles_interest', help='The interest graph directory.')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--simplified_ig_dir', type=str, default='interest/simplified_cles_interest',
                        help='The simplified interest graph directory.')

    args = parser.parse_args()

    # check dir
    check_dir(args)

    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")

    vocab = Vocab(VOCAL_FILE, args.vocab_size)

    # FILEWORDS
    FILTERWORD = []
    chinese_stopwords_path = os.getcwd() + r'''/stopwords/hit_stopwords.txt'''
    with open(chinese_stopwords_path, mode='r', encoding='utf-8') as f_stop:
        for line in f_stop:
            FILTERWORD.append(line.strip())
    punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'',
                    '`', '``',
                    '-', '--', '|', '\/']
    for a_punctuation in punctuations:
        if a_punctuation not in FILTERWORD:
            FILTERWORD.append(a_punctuation)

    filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
    filterids.append(vocab.word2id("[PAD]"))

    tfidf_w = readText(FILTER_WORD)

    lowtfidf_num = 0
    for w in tfidf_w:
        if vocab.word2id(w) != vocab.word2id('[UNK]'):
            FILTERWORD.append(w)
            filterids.append(vocab.word2id(w))
            lowtfidf_num += 1
        if lowtfidf_num > 5000:
            break

    igfiles = ['test_interest.json']
    datafiles = ['test.label.jsonl']

    for igfile, datafile in zip(igfiles, datafiles):
        cal_lost(os.path.join(args.ig_dir, igfile),
                    os.path.join(args.data_dir, datafile),
                 os.path.join(args.simplified_ig_dir, igfile + 'l' ),
                    FILTERWORD)


if __name__ == '__main__':
    main()