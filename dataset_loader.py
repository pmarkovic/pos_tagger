import argparse
import logging
from pprint import pprint

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


if __name__ == '__main__':
    args = arg_parser()
    dataset = datasets.load_dataset('dataset_loader.py', data_files={'train':args.train, 'test': args.test})
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased') # With AutoTokenizer, huggingface uses faster rust implementation of the tokenizer
    model = AutoModel.from_pretrained('bert-base-multilingual-cased')

    batch_wordlists = []
    for sent in dataset['train']:
        batch_wordlists.append(sent['word_list'])

    # offset_map returns start and end positions of bert tokens (0,0) for special tokens
    # https://stackoverflow.com/questions/66666525/how-to-map-token-indices-from-the-squad-data-to-tokens-from-bert-tokenizer
    encoded_input_with_offset_mapping = tokenizer(batch_wordlists, is_split_into_words=True, padding=True, return_tensors='pt', return_offsets_mapping=True)

    encoded_input = encoded_input_with_offset_mapping
    encoded_input.pop('offset_mapping')
    logging.info(encoded_input.keys(), encoded_input_with_offset_mapping.keys())

    embeddings = model(**encoded_input)[0].squeeze() # MemoryError on CPU - TODO: test on collab







