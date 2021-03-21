import argparse
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import pickle as pkl

import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModel

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import datasets

_DESCRIPTION = """\
 Every example in the dataset contains:
 1) a sentence ID
 2) word_list of the words in the sentence
 3) tag_list containing POS tag corresponding to each word 
"""

class PosData(datasets.GeneratorBasedBuilder):
    """
    converts .tsv files into hugging-face dataset. English dataset is split into train and validation. (80-20 split)
    srb-de-combined-file is loaded as the test dataset
    """

    def _info(self):
        return datasets.DatasetInfo(description=_DESCRIPTION,
                                    features=datasets.Features({
                                        "sent_id": datasets.Value("int32"),
                                        "word_list": datasets.features.Sequence(datasets.Value("string")),
                                        "tag_list": datasets.features.Sequence(datasets.Value("string")),
                                    })
                                    )

    def _split_generators(self, dl_manager):
        train_file, test_file = self.config.data_files['train'], self.config.data_files['test']

        # combine words into sequence_lists s.t. dataset = [(wordlist1, taglist1), (wordList2, tagList2)..]
        train_dataset = self.generate_sequenceBased_dataset(train_file)
        test_dataset = self.generate_sequenceBased_dataset(test_file)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"data": train_dataset[:int(0.8*len(train_dataset))]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"data": train_dataset[int(0.8*len(train_dataset)):]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"data": test_dataset})
        ]

    def generate_sequenceBased_dataset(self, filename):
        dataset = []
        with open(filename) as f:
            sent = []
            pos_tags = []
            for line in f:
                if line.startswith('*'):
                    # marks the end of line
                    #logging.info(f"{sent}\n{pos_tags}")
                    dataset.append((sent, pos_tags))
                    sent, pos_tags = [], []
                else:
                    tokens = line.split()
                    sent.append(tokens[1])
                    pos_tags.append(tokens[2])
        return dataset

    def _generate_examples(self, data):
        logging.info("generating examples..")
        for id, tup in enumerate(data):
            #logging.info(f"{tup[0]}\n{tup[1]}")
            yield id, {
                "sent_id": id,
                "word_list": tup[0],
                "tag_list": tup[1],
            }



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="test_data/ontonotes-4.0.tsv",
                        help='use this option to provide the path to English POS-tagged data')
    parser.add_argument("--test", default="test_data/test_set.tsv",
                        help="use this option to provide the path to combined srb-de POS-tagged data")
    args = parser.parse_args()
    return args


def create_input_batches(dataset, batch_size):
    batched_wordLists = []
    input_batches = []
    pos_tags = []
    for sent in dataset['train']:
        batched_wordLists.append(sent['word_list'])
        pos_tags.append(sent['tag_list'])
        if len(batched_wordLists) % batch_size == 0:
            input_batches.append((batched_wordLists, pos_tags))
            batched_wordLists = []
            pos_tags = []
    input_batches.append((batched_wordLists, pos_tags))
    return input_batches


if __name__ == '__main__':
    args = arg_parser()
    dataset = datasets.load_dataset('dataset_loader.py', data_files={'train':args.train, 'test': args.test})
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased') # With AutoTokenizer, huggingface uses faster rust implementation of the tokenizer
    model = AutoModel.from_pretrained('bert-base-multilingual-cased')

    input_batches = create_input_batches(dataset, batch_size=50)
    print(len(input_batches))
    Path("./temp_data").mkdir(parents=True, exist_ok=True)

    # generate embeddings
    for batched_wordLists, batched_ptags in input_batches:
        # offset_map returns (char_start, char_end) for each token. (0,0) for special tokens
        # https://stackoverflow.com/questions/66666525/how-to-map-token-indices-from-the-squad-data-to-tokens-from-bert-tokenizer
        encoded_input_with_offset_mapping = tokenizer(batched_wordLists, is_split_into_words=True, padding=True, return_tensors='pt', return_offsets_mapping=True)
        encoded_input = deepcopy(encoded_input_with_offset_mapping)
        encoded_input.pop('offset_mapping')
        # logging.info(f"encoded_input.keys() : {encoded_input.keys()}, encoded_input_with_offset_mapping.keys() {encoded_input_with_offset_mapping.keys()}")

        batched_word2token_position = [] # for every wordlist in the batch, stores [(word1, pos_tag1, token_start_pos, token_end_pos),..]

        for i, wordlist in enumerate(batched_wordLists):
            #print(wordlist)
            word2token_position = []
            k = 1
            for j, word in enumerate(wordlist):
                tup = encoded_input_with_offset_mapping['offset_mapping'][i][k]
                start_pos = k
                tup_len = tup[1] - tup[0]
                if len(word) == tup_len:
                    end_pos = k
                else: # iterate over the following tuples
                    while len(word) != tup_len:
                        k += 1
                        tup = encoded_input_with_offset_mapping['offset_mapping'][i][k]
                        tup_len += tup[1] - tup[0]
                    end_pos = k
                word2token_position.append((word, batched_ptags[i][j], start_pos, end_pos + 1)) # (word, pos_tag, token_start_pos, token_end_pos)
                k += 1
            #print(word2token_position)
            batched_word2token_position.append(word2token_position)

        embeddings = model(**encoded_input)[0].squeeze()
        #print(embeddings)
        # Create dict where keys are POS tags, and values are lists of embeddings
        # of all words that we encountered in dataset
        # Note for one POS tag we can have many embeddings for the same words
        # That can be good if these embeddings are different due to different contexts
        # But we can restrict that to include only one embeddings per word per POS tag
        tokens_embed = defaultdict(list)
        for ind, word2token_position in enumerate(batched_word2token_position):
            for tup in word2token_position:
                pos_tag = tup[1]
                embed = torch.mean(embeddings[ind][tup[2]:tup[3]], dim=0)
                tokens_embed[pos_tag].append(embed)
                #print(pos_tag, embed)
                with open('temp_data/ptag_embed_batch'+str(ind)+'.pkl', 'wb') as f:
                    pkl.dump(tokens_embed, f)








