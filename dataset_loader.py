import argparse
import logging
import torch
from transformers import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import datasets

_DESCRIPTION = """\
 Every example in the dataset contains:
 1) a sentence ID
 2) word_list of the words in the sentence
 3) tag_list containing POS tag corresponding to each word 
"""

class PosData(datasets.GeneratorBasedBuilder):

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    print(dataset['train'])
    # for sent in dataset['train']:
    #     print(sent['sentence'])
    encoded_dataset = dataset['train'].map(lambda sent: tokenizer(sent['word_list'], is_split_into_words=True, padding=True, return_tensors='pt'))
    print(encoded_dataset.column_names)
    print(encoded_dataset[0])
    #embeddings = model(**encoded_dataset)
    for ids in encoded_dataset['input_ids']:
        print(ids)
        print(tokenizer.decode(ids[0]))

    #TODO: convert to Torch dataset, https://colab.research.google.com/github/huggingface/datasets/blob/master/notebooks/Overview.ipynb#scrollTo=QvExTIZWvSVw
    # Format dataset to outputs torch.Tensor to pass through BertModel and train a pytorch model
    #columns = ['input_ids', 'token_type_ids', 'attention_mask',]
    #encoded_dataset.set_format(type='torch', columns=columns)


    # Instantiate a PyTorch Dataloader around our dataset
    # dynamic batching (pad on the fly with our own collate_fn)
    #def collate_fn(examples):
    #    return tokenizer.pad(examples, return_tensors='pt')


    #dataloader = torch.utils.data.DataLoader(encoded_dataset['train'], collate_fn=collate_fn, batch_size=8)
    #print(dataset.map(lambda example: print(len(example.train.features["word_list"]))))





