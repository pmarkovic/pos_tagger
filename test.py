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

    parser.add_argument("--model", default="./models/model.pt",
                        help="path to the trained model (default=./models/model.pt).")
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
    print("Obtaining pos tags embeddings...")
    tags_embed, tag2ind = get_pos_embeddings(args)

    print("Obtaining test set embeddings...")
    words_embed, labels = get_words_embeddings(args, tag2ind)

    print("Loading model...")
    model = torch.load(args.model)

    # Do prediction
    print("Doing predictions...")
    _, predictions = model(tags_embed, words_embed)

    # Evaluate predictions
    print("Evaluating predictions...")
    pred_labels = torch.argmax(predictions, dim=1)
    result = pred_labels == labels
    
    return result.sum().item() / result.shape[0]


if __name__ == "__main__":
    args = arg_parser()

    print(f"Start testing on set: {args.test}")
    
    test_result = test(args)

    print(f"Testing result: {test_result}!")
