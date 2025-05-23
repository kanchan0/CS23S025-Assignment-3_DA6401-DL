import argparse
import wandb
import torch
import pandas as pd
from model_utils import *

def main(args):
    # Initialize wandb
    wandb.init(project=args.wandb_project, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CSVs
    train_data = pd.read_csv(args.train_csv, header=None)
    val_data = pd.read_csv(args.val_csv, header=None)
    test_data = pd.read_csv(args.test_csv, header=None)

    train_input, train_output = train_data[0], train_data[1]
    val_input, val_output = val_data[0], val_data[1]
    test_input, test_output = test_data[0], test_data[1]

    # Continue with tokenization, dataset creation, model training etc.
    # This assumes the original notebook logic is re-used and modularized in model_utils.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="assignment3")

    # Add all extracted wandb.config values here manually
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beam_size", type=int, default=3)

    args = parser.parse_args()
    main(args)
