import os
import argparse

from dgl.data.utils import  save_graphs
from module.dataloader import ExampleSet
from module.vocabulary import Vocab

def main():
    parser = argparse.ArgumentParser(description='data2dgl.py')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CLES_word_chinese', help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/CLES_word_chinese',
                        help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str,
                        default='word_vector_file/chinese_merge_SGNS/word+chr+Ngram____merge_sgns_bigram_char300.txt',
                        help=' Path expression to external word embedding.')

    # Interstingpeparamter
    parser.add_argument('--use_interest', action='store_true', default=False, help='whether to use Interest Graph')
    parser.add_argument('--interest_dir', type=str, default='./aa_interest/simplified_cles_interest',
                        help='The Interest Graph Data')
    parser.add_argument('--pyrouge_temp_dir', type=str, default='./HSG+cles+ig/bb_pyrouge_temp',
                        help='The file to calculate score')
    parser.add_argument('--language', type=str, default='chinese', help='The chinese stopwords file')

    # Attention
    parser.add_argument('--attention', action='store_true', default=False, help='whether to use attention')

    # numworkers
    parser.add_argument('--num_workers', type=int, default=1, help='num of dataloader. [default: 1]')

    # Important settings
    parser.add_argument('--model', type=str, default='HSG', help='model structure[HSG|HDSG]')
    parser.add_argument('--restore_model', type=str, default='None',
                        help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='models\HSG_cles', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='logs/HSG_cles', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=True, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    parser.add_argument('--word_embedding', action='store_true', default=True,
                        help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False,
                        help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='GAT Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='lstm Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='whether to use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,
                        help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,
                        help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,
                        help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True,
                        help='use orthnormal init for lstm [default: True]')
    parser.add_argument('--sent_max_len', type=int, default=100,
                        help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50,
                        help='max length of documents (max timesteps of documents)')

    # Training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='for gradient clipping max gradient normalization')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')

    args = parser.parse_args()


    DATA_FILE = os.path.join(args.data_dir, "train.label.jsonl")
    VALID_FILE = os.path.join(args.data_dir, "val.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")

    vocab = Vocab(VOCAL_FILE, args.vocab_size)

    train_w2s_path = os.path.join(args.cache_dir, "train.w2s.tfidf.jsonl")
    val_w2s_path = os.path.join(args.cache_dir, "val.w2s.tfidf.jsonl")
    train_interest_path = os.path.join(args.interest_dir, 'train_interest.jsonl')
    val_interest_path = os.path.join(args.interest_dir, 'val_interest.jsonl')

    valid_dataset = ExampleSet(VALID_FILE, vocab, args.doc_max_timesteps, args.sent_max_len, FILTER_WORD,
                               val_w2s_path, args.use_interest, val_interest_path, args.language)
    print("Begin to tran data to dgl.")
    for G, i in valid_dataset:
        save_graphs('dgl_data/cles/valid/' + str(i) + '.graph.bin', [G])

if __name__ == '__main__':
    main()