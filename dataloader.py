import logging

import datasets
import torch
import numpy as np
import os
from transformers import BertTokenizer
from gen_embeddings import *

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


def eval(model, split='validation'):
    """split : accepts 'train' and 'validation' as values """
    args = arg_parser()  # arg_parser of gen_embeddings.py

    with open(f'{args.temp_dir}/ptag2embedding.pkl', 'rb') as f:
        ptag2embed_dict = pkl.load(f)

    if split == 'validation':
        files = os.listdir(args.examples_dir)[-2:] # last 2 files are for validation
    else:
        files = os.listdir(args.examples_dir)[:-2]

    all_results = []

    for filename in files:
        with open(filename, 'rb') as f:
            tag2wd_emb = pkl.load(f)

        words_embd = []
        labels = []

        tags = list(tag2wd_emb.keys())

        for ind, tag in enumerate(tags):
            for word in tag2wd_emb[tag]:
                words_embd.append(word)
                labels.append(ind)

        # Tokenize an episode examples
        #tags_ind = self.tokenizer(self.tags, return_tensors="pt", padding=True)
        #words_ind = self.tokenizer(words, return_tensors="pt", padding=True)
        tags_embd = [ptag2embed_dict[tag] for tag in tags]
        labels = torch.tensor(labels)

        _, predictions = model(tags_embd, words_embd)

        pred_labels = torch.argmax(predictions, dim=1)

        result = pred_labels == labels

        #return result.sum().item() / result.shape[0], pred_labels #TODO: check if it'S necessary to return pred_labels
        all_results.append(result.sum().item() / result.shape[0])

    # TODO: averge the results - not sure about what shape 'result.sum().item() / result.shape[0]' returns, so leaving for now
    return all_results


def get_episode_data(k, n):
    args = arg_parser() # arg_parser of gen_embeddings.py

    with open(f'{args.temp_dir}/ptag2embedding.pkl', 'rb') as f:
        ptag2embed_dict = pkl.load(f)

    filename = np.random.choice(os.listdir(args.examples_dir)[:-2], size=1)[0] # only first 10 files are for training

    with open(f'{args.examples_dir}/{filename}', 'rb') as f:
        tag2wd_emb = pkl.load(f)

    # Select tags (ways) for an episode
    tags = list(tag2wd_emb.keys())
    #logging.info(f'choose tags from: {tags}')
    Nc = np.random.choice(range(len(tags)), size=k, replace=False)
    episode_tags = [tags[c] for c in Nc]

    # Select words (shots) for an episode and corresponding labels
    episode_word_embs = []
    episode_labels = [] # label pertaining to each word in episode_word_embs
    for ind, tag in enumerate(episode_tags):
        for word in np.random.choice(tag2wd_emb[tag], size=n, replace=False):
            episode_word_embs.append(word)
            episode_labels.append(ind)

    episode_tags_embs = [ ptag2embed_dict[tag] for tag in episode_tags ]  # embeddings for the randomly chosen tags/labels
    episode_tags_embs = torch.cat(episode_tags_embs).to('cpu') # device = 'cpu'/'gpu'
    episode_labels = torch.tensor(episode_labels)
    episode_word_embs = torch.stack(episode_word_embs).to('cpu')

    #print(f'episode_tags_embs {episode_tags_embs.shape} episode_word_embs {episode_word_embs.shape}')

    return  episode_tags_embs, episode_word_embs, episode_labels


# class DataLoader:
#     """
#     Implementation of class for working with data.
#     """

    # def __init__(self, bert_model, k, n):
    #     # model must be downloaded locally in order for this to work
    #     # in Google Colab works fine
    #     self.tokenizer = BertTokenizer(bert_model)
    #
    #     self.k = k
    #     self.n = n
    #
    #     # TODO
    #     # This is just my toy example, should be removed :D
    #     self.tags = ["VB", "NN", "JJ", "RB", "DT"]
    #     self.examples = {"VB": ["running", "sleeping", "liked", "throw", "spoken", "buying", "bought", "slept", "dreaming", "won"],
    #                      "NN": ["house", "ball", "worker", "tiger", "student", "class", "water", "bag", "car", "drinker"],
    #                      "JJ": ["nice", "bad", "blue", "cold", "refreshing", "lazy", "noisy", "clear", "compact", "complex"],
    #                      "RB": ["slowly", "warmly", "diligently", "sadly", "normally", "daily", "gently", "quite", "regularly", "now"],
    #                      "DT": ["a", "the", "many", "much", "these", "that", "our", "yours", "ten", "other"]}

    


    # TODO
    # This method probably should be in some other place, 
    # but it was convenient to put it here now :D 
    # def eval(self, model):
    #     words = []
    #     labels = []
    #
    #     for ind, tag in enumerate(FILTERED_POS_TAGS):
    #         for word in self.examples[tag]:
    #             words.append(word)
    #             labels.append(ind)
    #
    #     # Tokenize an episode examples
    #     tags_ind = self.tokenizer(self.tags, return_tensors="pt", padding=True)
    #     words_ind = self.tokenizer(words, return_tensors="pt", padding=True)
    #     labels = torch.tensor(labels)
    #
    #     _, predictions = model(tags_ind, words_ind)
    #
    #     pred_labels = torch.argmax(predictions, dim=1)
    #
    #     result = pred_labels == labels
    #
    #     return result.sum().item() / result.shape[0], pred_labels


