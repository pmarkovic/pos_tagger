import torch
import numpy as np
from transformers import BertTokenizer


class DataLoader:
    """
    Implementation of class for working with data.
    """

    def __init__(self, bert_model, k, n):
        # model must be downloaded locally in order for this to work
        # in Google Colab works fine
        self.tokenizer = BertTokenizer(bert_model)

        self.k = k
        self.n = n

        # TODO 
        # This is just my toy example, should be removed :D
        self.tags = ["VB", "NN", "JJ", "RB", "DT"]
        self.examples = {"VB": ["running", "sleeping", "liked", "throw", "spoken", "buying", "bought", "slept", "dreaming", "won"],
                         "NN": ["house", "ball", "worker", "tiger", "student", "class", "water", "bag", "car", "drinker"],
                         "JJ": ["nice", "bad", "blue", "cold", "refreshing", "lazy", "noisy", "clear", "compact", "complex"],
                         "RB": ["slowly", "warmly", "diligently", "sadly", "normally", "daily", "gently", "quite", "regularly", "now"],
                         "DT": ["a", "the", "many", "much", "these", "that", "our", "yours", "ten", "other"]} 

    def get_episode_data(self):
        # Select tags (ways) for an episode
        Nc = np.random.choice(range(len(self.tags)), size=self.k, replace=False)
        episode_tags = [self.tags[c] for c in Nc]

        # Select words (shots) for an episode and corresponding labels
        episode_words = []
        episode_labels = []
        for ind, tag in enumerate(episode_tags):
            for word in np.random.choice(self.examples[tag], size=self.n, replace=False):
                episode_words.append(word)
                episode_labels.append(ind)

        # Tokenize an episode examples
        episode_tags_ind = self.tokenizer(episode_tags, return_tensors="pt", padding=True)
        episode_words_ind = self.tokenizer(episode_words, return_tensors="pt", padding=True)
        episode_labels = torch.tensor(episode_labels)

        return episode_tags_ind, episode_words_ind, episode_labels


    # TODO
    # This method probably should be in some other place, 
    # but it was convenient to put it here now :D 
    def eval(self, model):
        words = []
        labels = []

        for ind, tag in enumerate(self.tags):
            for word in self.examples[tag]:
                words.append(word)
                labels.append(ind)
        
        # Tokenize an episode examples
        tags_ind = self.tokenizer(self.tags, return_tensors="pt", padding=True)
        words_ind = self.tokenizer(words, return_tensors="pt", padding=True)
        labels = torch.tensor(labels)

        _, predictions = model(tags_ind, words_ind)

        pred_labels = torch.argmax(predictions, dim=1)

        result = pred_labels == labels

        return result.sum().item() / result.shape[0], pred_labels