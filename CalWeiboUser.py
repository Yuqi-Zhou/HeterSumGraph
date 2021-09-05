import os
import json
import datetime
import argparse

import pandas as pd


def getfiles(datadir):
    for root, dirs, files in os.walk('.' + datadir):
        for i, file in enumerate(files):
            files[i] = os.path.join(root, file)
        print(files[0])
        return files


def getdata(files):
    data = []
    for file in files:
        data.append(pd.read_csv(file))
    return data

def main():
    parser = argparse.ArgumentParser(description='CalWeiboUser.py')

    parser.add_argument('--data_dir', type=str, default='/data/weibo', help='the data dir')
    parser.add_argument('--save_dir', type=str, default='example/cal_weibo', help='the save dir')

    args = parser.parse_args()

    files = getfiles(args.data_dir)
    data = getdata(files)   # data:DataFrame

    dicts = {}

    for i, lines in enumerate(data):
        for line in lines.itertuples():
                id = line.__getattribute__('源微博id')
                if id in dicts.keys():
                    dicts[id]['转发次数'] += 1
                    dicts[id]['转发者'].append(files[i].split('/')[-1].replace('.csv', ''))
                else:
                    dicts[id] = {'正文': line.__getattribute__('源微博文章正文'), '转发次数': 1,
                                 '转发者':[files[i].split('/')[-1].replace('.csv', '')],
                                 '摘要': line.__getattribute__('源微博正文')}

    newdicts = [dict for dict in dicts.values() if dict['转发次数'] >= 2]
    counts = {}
    total = 0
    for dict in dicts.values():
        if dict['转发次数'] in counts.keys():
            counts[dict['转发次数']] += 1
        else:
            counts[dict['转发次数']] = 1
        total += dict['转发次数']
    print("平均每篇文章被转发{}次".format(total/len(dicts)))

    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    SAVE_PATH = os.path.join(args.save_dir, nowTime+ 'filter' +'.json')
    SAVE_PATH2 = os.path.join(args.save_dir, nowTime + 'count' + '.json')
    SAVE_PATH3 = os.path.join(args.save_dir, nowTime + 'total' + '.json')
    with open(SAVE_PATH, 'w') as f:
        json.dump(newdicts, f, ensure_ascii=False, indent=1)
    with open(SAVE_PATH2, 'w') as f:
        json.dump(counts, f, ensure_ascii=False, indent=1)
    with open(SAVE_PATH3, 'w') as f:
        json.dump(dicts, f, ensure_ascii=False, indent=1)
    exit(0)


if __name__ == '__main__':
    main()