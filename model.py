import torch.nn as nn
from transformers import BertModel


class ProtoNet(nn.Module):
    """
    Implementation of Prototypical network model.
    """
  
    def __init__(self, bert_model, mdim, bdim):
        super(ProtoNet, self).__init__()
        
        # model must be downloaded locally in order for this to work
        # in Google Colab works fine
        self.embed = BertModel.from_pretrained(bert_model)

        self.tags_linear = nn.Linear(bdim, mdim)
        self.words_linear = nn.Linear(bdim, mdim)

        for param in self.embed.parameters():
            param.requires_grad=False

    def forward(self, tags_ind, words_ind, prediction=True):
        # Obtain BERT embeddings
        tags_embed = self.embed(**tags_ind)[1]
        words_embed = self.embed(**words_ind)[1]

        # Transform to metric system
        tags_metric = self.tags_linear(tags_embed)
        words_metric = self.words_linear(words_embed)

        # To get embeddings instead of predictions
        # Should be useful for visualization
        if not prediction:
            return tags_metric, words_metric

        # output - distance
        distances = self._calc_distances(words_metric, tags_metric)

        # log softmax for loss calc
        log_pred_y = (-distances).log_softmax(dim=1)

        # prediction softmax
        pred_y = (-distances).softmax(dim=1)

        return log_pred_y, pred_y

    def _calc_distances(self, words, tags):
        n_words = words.shape[0]
        n_tags = tags.shape[0]
        
        # Calculate squared Euclidean distance
        distances = (
                words.unsqueeze(1).expand(n_words, n_tags, -1) -
                tags.unsqueeze(0).expand(n_words, n_tags, -1)
        ).pow(2).sum(dim=2)

        return distances

