import os
import json
import argparse
import jsonlines
import requests

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

wiki_version = "wiki_2019"
IP_ADDRESS = "http://localhost"
PORT = "1235"
base_url = "/home/yuqi/ex_sum/EL_data"

def example_preprocessing(texts):
    # user does some stuff, which results in the format below.
    processed = {}
    for i, sentence in enumerate(texts):
        processed["test_doc"+str(i+1)] = [texts[i], []]
    return processed

def main():
    parser = argparse.ArgumentParser("EL_CNNDM.py")

    parser.add_argument('--data_dir', type=str, default='data/CNNDM', help='the data directory.')
    parser.add_argument('--save_root', type=str, default='interest/CNNDM', help='the interest directory.')
    args = parser.parse_args()

    mention_detection = MentionDetection(base_url, wiki_version)
    tagger_ner = load_flair_ner("ner-fast")
    tagger_ngram = Cmns(base_url, wiki_version, n=5)

    config = {
        "mode": "eval",
        "model_path": os.path.join(base_url, "ed-wiki-2019/model"),
    }
    model = EntityDisambiguation(base_url, wiki_version, config)

    # get file dir
    files_dir = []
    for root, dir, files in os.walk(args.data_dir):
        for file in files:
            files_dir.append(os.path.join(root, file))

    # begin read jsonl
    for file in files_dir:
        INTEREST_FILE = os.path.join(args.save_root, file.split('/')[-1].replace('.label', ''))
        # print(INTEREST_FILE)
        with jsonlines.open(file, 'r') as f_r:
            with jsonlines.open(INTEREST_FILE, 'w') as f_w:
                for line in f_r:
                    texts = line['text']
                    result = {'text': texts,
                              'linking_entity': []}
                    input_text = example_preprocessing(texts)
                    mentions_dataset, n_mentions = mention_detection.find_mentions(input_text, tagger_ner)
                    predictions, timing = model.predict(mentions_dataset)
                    linking_entity = process_results(mentions_dataset, predictions, input_text)
                    result['linking_entity'] = linking_entity
                    jsonlines.Writer.write(f_w,result)


if __name__ == '__main__':
    main()
