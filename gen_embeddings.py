from collections import defaultdict

import torch
from transformers import BertModel, BertTokenizer


"""
A quick idea how we can have contextual embeddings for POS tags.
This will require of us to change logic so that model now
receives embeddings and do linear transformation to the metric space.
While embeddings will be obtained from dataloader.
Or some other approach of course.
"""


test_example = """0	John	NNP
1	V.	NNP
2	Holmes	NNP
3	,	,
4	an	DT
5	investment	NN
6	-	HYPH
7	newsletter	NN
8	publisher	NN
9	,	,
10	and	CC
11	three	CD
12	venture	NN
13	-	HYPH
14	capital	NN
15	firms	NNS
16	he	PRP
17	organized	VBD
18	were	VBD
19	enjoined	VBN
20	from	IN
21	violating	VBG
22	the	DT
23	registration	NN
24	provisions	NNS
25	of	IN
26	the	DT
27	securities	NNS
28	laws	NNS
29	governing	VBG
30	investment	NN
31	companies	NNS
32	.	."""


def generate_embeddings():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")

    sentence = []
    pos_tags = []

    for line in test_example.split("\n"):
        tokens = line.split("\t")
        sentence.append(tokens[1])
        pos_tags.append(tokens[2])


    # This part is required to determine position of full words
    # since a word can consists of multiple tokens (e.g. bonfire - b + ##on + ##fire)
    indices = tokenizer(sentence, is_split_into_words=True, return_tensors="pt")

    words_positions = []
    curr_token = ""
    prev_position = 1
    # First and last tokens are special tokens [CLS] and [SEP]
    for position, id in enumerate(indices["input_ids"].squeeze()[1:-1]):
        # Convert to word
        token_value = tokenizer.ids_to_tokens[int(id)]

        # Discard ## from the begining of a word if it is there
        if token_value.startswith("#"):
            token_value = token_value[2:]
        curr_token += token_value

        # If word is complete (added all parts of multi-token words) add it
        if curr_token == sentence[0]:
            # Need to add 2 because we discarded first position 
            words_positions.append((curr_token, prev_position, position+2))

            sentence.pop(0)
            curr_token = ""
            prev_position = position + 2

    # Generate embeddings for tokens
    embeddings = model(**indices)[0].squeeze()

    # Create dict where keys are POS tags, and values are lists of embeddings
    # of all words that we encountered in dataset
    # Note for one POS tag we can have many embeddings for the same words
    # That can be good if these embeddings are different due to different contexts
    # But we can restrict that to include only one embeddings per word per POS tag 
    tokens_embed = defaultdict(list)
    for ind, token in enumerate(words_positions):
        pos_tag = pos_tags[ind]
        embed = torch.mean(embeddings[token[1]:token[2]], dim=0)
        tokens_embed[pos_tag].append(embed)

    return tokens_embed


if __name__ == "__main__":
    generate_embeddings()
