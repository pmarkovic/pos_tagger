import argparse
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import pickle as pkl

import torch
import datasets
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

FILTERED_POS_TAGS = {"NNP", ",", "DT", "NN", "CC", "CD", "NNS", "PRP", "VBD", "VBN", 
                     "IN", "VBG", ".", "RB", "JJ", "TO", "VB", "VBP", "MD", "VBZ"}
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


def create_input_batches(dataset, batch_size, split='train'):
    """split: accepts the following arguments - 'train', 'test', 'validation'"""
    batched_wordLists = []
    input_batches = []
    pos_tags = []

    for sent in dataset[split]:
        batched_wordLists.append(sent['word_list'])
        pos_tags.append(sent['tag_list'])

        if len(batched_wordLists) % batch_size == 0:
            input_batches.append((batched_wordLists, pos_tags))
            batched_wordLists = []
            pos_tags = []

    input_batches.append((batched_wordLists, pos_tags))
    
    return input_batches


def save_ptag_embeddings(args, tokenizer, model):
    ptag2embedding = dict()

    for ptag in FILTERED_POS_TAGS:
        encoded_tag = tokenizer(ptag, return_tensors="pt", padding=True)
        embedding = model(**encoded_tag)[0].squeeze()
        ptag2embedding[ptag] = torch.mean(embedding, dim=0, keepdim=True)

    with open(f'{args.ptag_emb}/ptag2embedding.pkl', 'wb') as f:
        pkl.dump(ptag2embedding, f)

    logging.info(f"saved POS-tag embeddings to {args.ptag_emb}/ptag2embedding.pkl")


def generate_embeddings(args):
    """
    Generates embeddings for POS-tags and training examples (i.e. words in sentences)
    """
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ptag_emb).mkdir(parents=True, exist_ok=True)
    Path(args.examples_dir).mkdir(parents=True, exist_ok=True)

    # With AutoTokenizer, huggingface uses faster rust implementation of the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model)

    # create embeddings for POS-tags
    save_ptag_embeddings(args, tokenizer, model)

    dataset = datasets.load_dataset('gen_embeddings.py', data_files={'train': args.train, 'test': args.test})
    # create input batches for training
    input_batches = create_input_batches(dataset, batch_size=args.batch_size, split='train')
    logging.info(f"Number of input batches : {len(input_batches)}")

    n_batch = 0

    # generate embeddings
    for batched_wordLists, batched_ptags in input_batches:
        # offset_map returns (char_start, char_end) for each token. (0,0) for special tokens
        # https://stackoverflow.com/questions/66666525/how-to-map-token-indices-from-the-squad-data-to-tokens-from-bert-tokenizer
        encoded_input_with_offset_mapping = tokenizer(batched_wordLists, is_split_into_words=True, padding=True, return_tensors='pt', return_offsets_mapping=True)
        encoded_input = deepcopy(encoded_input_with_offset_mapping)
        encoded_input.pop('offset_mapping')

        batched_word2token_position = [] # for every wordlist in the batch, stores [(word1, pos_tag1, token_start_pos, token_end_pos),..]
        tokens_embed = defaultdict(list)

        for i, wordlist in enumerate(batched_wordLists):
            word2token_position = []
            k = 1 # skip 0th position that contains special token [CLS]
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
            batched_word2token_position.append(word2token_position)

        embeddings = model(**encoded_input)[0].squeeze()

        # Create dict where keys are POS tags, and values are lists of embeddings
        # of all words that we encountered in dataset
        # Note for one POS tag we can have many embeddings for the same words
        # That can be good if these embeddings are different due to different contexts
        # But we can restrict that to include only one embeddings per word per POS tag
        for ind, word2token_position in enumerate(batched_word2token_position):
            for tup in word2token_position:
                pos_tag = tup[1]
                embed = torch.mean(embeddings[ind][tup[2]:tup[3]], dim=0)
                tokens_embed[pos_tag].append(embed)
                #print(pos_tag)

        with open(f"{args.temp_dir}/batch_{n_batch}_embeds.pkl", "wb") as f:
            pkl.dump(tokens_embed, f)

        n_batch += 1
        if n_batch % 100 == 0:
            logging.info(f"{n_batch} batches processed...")


def merge_files(args):
    files_path = Path(args.temp_dir)
    final_embed = defaultdict(list)

    for ind, file in enumerate(files_path.iterdir()):
        if file.is_file():
            if ind % 100 == 0:
                with open(f"{args.examples_dir}/examples_{ind // 100}.pkl", "wb") as f:
                    pkl.dump(final_embed, f)
                    final_embed.clear()

            with file.open("rb") as f:
                curr_embed = pkl.load(f)
            
            for key, embeddings in curr_embed.items():
                if key not in FILTERED_POS_TAGS:
                    continue

                for embed in embeddings:
                    final_embed[key].append(embed)

    with open(f"{args.examples_dir}/examples_{ind // 100 + 1}.pkl", "wb") as f:
        pkl.dump(final_embed, f)
        final_embed.clear()

    logging.info(f"saved embeddings for the final training examples to {args.examples_dir}")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="test_data/ontonotes-4.0.tsv",
                        help='use this option to provide the path to English POS-tagged data')
    parser.add_argument("--test", default="test_data/srb_test_set.tsv",
                        help="use this option to provide the path to SRB or DE POS-tagged data")
    parser.add_argument("--temp_dir", default="./temp_data",
                        help="directory where to save intermediate train examples (default=./temp_data).")
    parser.add_argument("--examples_dir", default="./embeddings/train_examples",
                        help="directory where to save final train examples (default=./embeddings/train_examples).")
    parser.add_argument("--ptag_emb", default="./embeddings",
                        help='use this option to provide the path to pickled POS-tag embeddings (default: "./embeddings")')
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased",
                        help="bert model to use for encoding text (default=bert-base-multilingual-cased).")
    parser.add_argument("--gen_embed", default=False, action="store_true",
                        help="flag to indicate if embeddings should be generated (default=False).")
    parser.add_argument("--batch_size", default=50, type=int,
                        help="batches of sentences will be created with the given size for generating embeddings (default=50)")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    
    if args.gen_embed:
        generate_embeddings(args)

    merge_files(args)
