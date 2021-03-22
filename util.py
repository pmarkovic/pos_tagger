import torch
import torch.nn as nn
import numpy as np
import os
import pickle as pkl


def eval(model, ptag_emb_filepath, train_eg_dir, device, split='validation'):
    """
    This fn. performs evaluation over training/validation split.
    :param split : accepts 'train' and 'validation' as values
    """

    with open(ptag_emb_filepath, 'rb') as f:
        ptag2embed_dict = pkl.load(f)

    if split == 'validation':
        files = sorted(os.listdir(train_eg_dir))[-2:] # last 2 files are for validation
    else:
        files = sorted(os.listdir(train_eg_dir))[:-2]

    all_results = []

    loss_fn = nn.NLLLoss()
    total_loss = 0
    total_labels = 0

    for filename in files:
        with open(os.path.join(train_eg_dir, filename), 'rb') as f:
            tag2wd_emb = pkl.load(f)

        words_embd = []
        labels = []

        tags = list(tag2wd_emb.keys())

        for ind, tag in enumerate(tags):
            for word in tag2wd_emb[tag]:
                words_embd.append(word)
                labels.append(ind)

        # Tokenize an episode examples
        tags_embd = [ptag2embed_dict[tag] for tag in tags]
        tags_embd = torch.cat(tags_embd).to(device)
        words_embd = torch.stack(words_embd).to(device)
        labels = torch.tensor(labels)

        log_loss, predictions = model(tags_embd, words_embd)

        # Calculate loss of the examples from the current file
        loss = loss_fn(log_loss, labels)
        total_loss += loss.item()
        total_labels += len(labels)

        # Calculate predictions
        pred_labels = torch.argmax(predictions, dim=1)
        result = pred_labels == labels
        all_results.append(result.sum().item() / result.shape[0])

    return np.mean(np.asarray(all_results)), total_loss / total_labels


def get_episode_data(ptag_emb_filepath, train_eg_dir, k, n, device='cpu'):
    """
    This fn. prepares input for episodic learning
    :param k: number of classes i.e. POS-tags
    :param n: number of examples
    :return:
    """

    with open(f'{ptag_emb_filepath}', 'rb') as f:
        ptag2embed_dict = pkl.load(f)

    filename = np.random.choice(sorted(os.listdir(train_eg_dir))[:-2], size=1)[0] # only first 10 files are for training

    with open(f'{train_eg_dir}/{filename}', 'rb') as f:
        tag2wd_emb = pkl.load(f)

    # Select tags (ways) for an episode
    tags = list(tag2wd_emb.keys())
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
    episode_tags_embs = torch.cat(episode_tags_embs).to(device)
    episode_labels = torch.tensor(episode_labels)
    episode_word_embs = torch.stack(episode_word_embs).to(device)

    return  episode_tags_embs, episode_word_embs, episode_labels


