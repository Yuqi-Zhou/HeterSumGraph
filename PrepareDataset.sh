#!/usr/bin/env bash
# 命令示例:     ./PrepareDataset.sh CLES_word_chinese aa_data_datasets/cles_word_chinese
# 要有三个参数, 或两个参数(task 不要).
dataset="$1"    # 数据集的名称, 用作图数据文件夹 cache 下的文件夹名, 如 dataset 为 'CLES_word'
datadir="$2"    # 原数据集目录的路径, 如 aa_data_datasets/cles_word
task="$3"


# -u to check bounded args, -e to exit when error
#set -u
#set -e
set -eux     # ------------------------------------ 我自己加的, 原先只有 set -e
set -o pipefail  # ------------------------------------ 我自己加的

if [ ! -n "$dataset" ]; then
    echo "lose the dataset name!"
    exit
fi

if [ ! -n "$datadir" ]; then
    echo "lose the data directory!"
    exit
fi

if [ ! -n "$task" ]; then
    task=single
fi

type=(train val test)

echo -e "\033[34m[Shell] Create Vocabulary! \033[0m"
#python script/createVoc.py --dataset $dataset --data_path $datadir/train.label.jsonl
python script/createVoc.py --dataset $dataset --data_path $datadir/train.label_含summary键.jsonl  # ------------------------------------ 我自己改动上一行的

echo -e "\033[34m[Shell] Get low tfidf words from training set! \033[0m"
python script/lowTFIDFWords.py --dataset $dataset --data_path $datadir/train.label.jsonl

echo -e "\033[34m[Shell] Get word2sent edge feature! \033[0m"
for i in ${type[*]}
    do
        python script/calw2sTFIDF.py --dataset $dataset --data_path $datadir/$i.label.jsonl
    done

if [ "$task" == "multi" ]; then
    echo -e "\033[34m[Shell] Get word2doc edge feature! \033[0m"
    for i in ${type[*]}
        do
            python script/calw2dTFIDF.py --dataset $dataset --data_path $datadir/$i.label.jsonl
        done
fi

echo -e "\033[34m[Shell] The preprocess of dataset $dataset has finished! \033[0m"


