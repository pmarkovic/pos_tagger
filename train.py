import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from util import get_episode_data, eval
from model import ProtoNet

def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default='models/model.pt',
                        help="path where to save the trained model (default=models/model.pt).")
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased",
                        help="bert model to use for encoding text (default=bert-base-multilingual-cased).")
    parser.add_argument("--mdim", default=512, type=int,
                        help="dimension of embeddings in a metric space (default=512).")
    parser.add_argument("--bdim", default=768, type=int,
                        help="dimension of bert model embeddings (default=768).")
    parser.add_argument("--seed", default=777, type=int,
                        help="seed for reproducibility purpose (default=777).")
    parser.add_argument("--k", default=10, type=int,
                        help="number of classes/tags per episode (default=10).")
    parser.add_argument("--n", default=5, type=int,
                        help="number of examples/shots per class per episode (default=5).")
    parser.add_argument("--ep", default=100, type=int,
                        help="number of episodes per training (default=100).")
    parser.add_argument("--lr", default=1e-3,
                        help="learning rate (default=1e-3).")
    parser.add_argument("--train", default="./embeddings/train_examples",
                        help='use this option to provide the path to pickled training examples that were generated by gen_embeddings.py'
                             'default: ./embeddings/train_examples')
    parser.add_argument("--test", default="test_data/test_set.tsv",
                        help="use this option to provide the path to the combined srb-de POS-tagged data")
    parser.add_argument("--ptag_emb", default="./embeddings/ptag2embedding.pkl",
                        help='use this option to provide the path to pickled POS-tag embeddings (default: "./embeddings/ptag2embedding.pkl")')
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
    model = ProtoNet(args.mdim, args.bdim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.NLLLoss()

    print(f"Device: {device}")
    print(f"Hyperparameters:\n\tK: {args.k}\n\tN: {args.n}\n\tEp: {args.ep}\n\tlr: {args.lr}")
    print(f"{''.join(['-']*20)}\n")

    for episode in range(args.ep):
        optimizer.zero_grad()

        # Select the episode examples
        tags_embd, words_embd, true_labels = get_episode_data(args.ptag_emb, args.train, args.k, args.n, device)

        # Episode pass
        log_loss, _ = model(tags_embd, words_embd)

        # Calc loss
        loss = loss_fn(log_loss, true_labels)
        print(f"Episode {episode+1} loss: {loss.item()}")
        
        # Calcualte train and validation losses after 10 episodes
        if (episode+1) % 10 == 0:
            _, train_loss = eval(model, args.ptag_emb, args.train, device, split="train")
            _, valid_loss = eval(model, args.ptag_emb, args.train, device)

            print(f"Train loss {train_loss}, after {episode+1} episodes.")
            print(f"Validation loss {valid_loss}, after {episode+1} episodes.")

        # Backprop step
        loss.backward()
        optimizer.step()

    print("End of training...")

    torch.save(model, args.model_path)


if __name__ == "__main__":
    args = arg_parser()

    train(args)

