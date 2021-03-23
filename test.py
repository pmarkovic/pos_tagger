import os
import json
import argparse
import pickle as pkl
from copy import deepcopy
from collections import defaultdict

import torch
import datasets
from transformers import AutoTokenizer, AutoModel

from gen_embeddings import create_input_batches, FILTERED_POS_TAGS



def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--models_dir", default="./models",
                        help="path to the trained models (default=./models).")
    parser.add_argument("--results_dir", default="./results",
                        help="path to the results dir (default=./results).")
    parser.add_argument("--test", default="./test_data/de-test.tsv",
                        help="path to the test set (default=./test_data/de-test.tsv).")
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased",
                        help="bert model to use for encoding text (default=bert-base-multilingual-cased).")
    parser.add_argument("--ptag_emb", default="./embeddings/ptag2embedding.pkl",
                        help='use this option to provide the path to pickled POS-tag embeddings (default: "./embeddings")')
    parser.add_argument("--batch_size", default=50, type=int,
                        help="batches of sentences will be created with the given size for generating embeddings (default=50)")

    args = parser.parse_args()

    return args


def get_words_embeddings(args, tag2ind):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model)

    dataset = datasets.load_dataset('gen_embeddings.py', data_files={'train': "./data/ontonetes-4.0.tsv", 'test': args.test})
    input_batches = create_input_batches(dataset, batch_size=args.batch_size, split='test')

    # generate embeddings
    words_embd = []
    labels = []
    for batched_wordLists, batched_ptags in input_batches:
        encoded_input_with_offset_mapping = tokenizer(batched_wordLists, 
                                                      is_split_into_words=True, 
                                                      padding=True, 
                                                      return_tensors='pt', 
                                                      return_offsets_mapping=True)
        encoded_input = deepcopy(encoded_input_with_offset_mapping)
        encoded_input.pop('offset_mapping')

        batched_word2token_position = []
        for i, wordlist in enumerate(batched_wordLists):
            word2token_position = []
            k = 1
            for j, word in enumerate(wordlist):
                tup = encoded_input_with_offset_mapping['offset_mapping'][i][k]
                start_pos = k
                tup_len = tup[1] - tup[0]
                if len(word) == tup_len:
                    end_pos = k
                else:
                    while len(word) != tup_len:
                        k += 1
                        tup = encoded_input_with_offset_mapping['offset_mapping'][i][k]
                        tup_len += tup[1] - tup[0]
                    end_pos = k
                word2token_position.append((word, batched_ptags[i][j], start_pos, end_pos + 1))
                k += 1
            batched_word2token_position.append(word2token_position)

        embeddings = model(**encoded_input)[0].squeeze()
        
        for ind, word2token_position in enumerate(batched_word2token_position):
            for tup in word2token_position:
                pos_tag_ind = tag2ind[tup[1]]
                embed = torch.mean(embeddings[ind][tup[2]:tup[3]], dim=0)
                words_embd.append(embed)
                labels.append(pos_tag_ind)

    words_embd = torch.stack(words_embd)
    labels = torch.tensor(labels)

    return words_embd, labels


def get_pos_embeddings(args):
    tags_embd = []
    tag2ind = defaultdict(int)
    
    with open(args.ptag_emb, 'rb') as f:
        ptag2embed_dict = pkl.load(f)

    for ind, tag in enumerate(FILTERED_POS_TAGS):
        tags_embd.append(ptag2embed_dict[tag])
        tag2ind[tag] = ind
    
    tags_embd = torch.cat(tags_embd)

    return tags_embd, tag2ind


def test(args):
    results = defaultdict(float)
    tags_embed, tag2ind = get_pos_embeddings(args)
    words_embed, labels = get_words_embeddings(args, tag2ind)

    for model_name in os.listdir(args.models_dir):
        model_path = os.path.join(args.models_dir, model_name)
        model = torch.load(model_path)
        model_result = 0
        
        for i in range(10):
            _, predictions = model(tags_embed, words_embed)
            pred_labels = torch.argmax(predictions, dim=1)
            curr_result = pred_labels == labels
            model_result += curr_result.sum().item() / curr_result.shape[0]

        results[model_name.split('.')[0]] = round(model_result / 10, 2)
    
    return results


def save_results(args, results):
    test_set_name = args.test[:-4].split('/')[-1]
    results_file = os.path.join(args.results_dir, f"{test_set_name}.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = arg_parser()

    results = test(args)

    save_results(args, results)
