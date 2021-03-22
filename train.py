import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataloader import get_episode_data
from model import ProtoNet

def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default='models/model.pt',
                        help="path where to save the trained model (default=models/model.pt).")
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased",
                        help="bert model to use for encoding text (default=bert-base-multilingual-cased).")
    parser.add_argument("--mdim", default=512,
                        help="dimension of embeddings in a metric space (default=512).")
    parser.add_argument("--bdim", default=768,
                        help="dimension of bert model embeddings (default=768).")
    parser.add_argument("--seed", default=777,
                        help="seed for reproducibility purpose (default=777).")
    parser.add_argument("--k", default=10,
                        help="number of classes/tags per episode (default=10).")
    parser.add_argument("--n", default=5,
                        help="number of examples/shots per class per episode (default=5).")
    parser.add_argument("--ep", default=100,
                        help="number of episodes per training (default=100).")
    parser.add_argument("--lr", default=1e-3,
                        help="learning rate (default=1e-3).")

    args = parser.parse_args()
    return args


def train(args):
    # For the reproducibility purpose
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Initialization
    print("Initialization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #dataloader = DataLoader(args.bert_model, args.k, args.n)
    model = ProtoNet(args.bert_model, args.mdim, args.bdim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.NLLLoss()

    print(f"Device: {device}")
    print(f"Hyperparameters:\n\tK: {args.k}\n\tN: {args.n}\n\tEp: {args.ep}\n\tlr: {args.lr}")
    print(f"{''.join(['-']*20)}\n")

    cycle_loss = 0

    for episode in range(args.ep):
        #print(f"Starting episode: {episode+1}...")
        optimizer.zero_grad()

        # Select the episode examples
        tags_embd, words_embd, true_labels = get_episode_data(args.k, args.n)

        # Episode pass
        log_loss, predictions = model(tags_embd, words_embd)

        # Calc loss
        loss = loss_fn(log_loss, true_labels)
        cycle_loss += loss.item()
        
        # TODO
        # This should be change, so that we evaluate model after N episodes
        # both on train and validation sets.
        # But we should also keep track of loss per episode
        if (episode+1) % 10 == 0:
            print(f"After {episode+1} episodes, loss: {cycle_loss / 10}")
            cycle_loss = 0

        # Backprop step
        loss.backward()
        optimizer.step()

    print("End of training...")

    torch.save(model, args.model_path)


if __name__ == "__main__":
    args = arg_parser()

    train(args)

