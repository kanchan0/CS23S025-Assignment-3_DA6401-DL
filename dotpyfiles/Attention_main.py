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

    tokenizer = SimpleTokenizer()
    vocab_size = 256

    train_dataset = TranslationDataset(train_input, train_output, tokenizer)
    val_dataset = TranslationDataset(val_input, val_output, tokenizer)
    test_dataset = TranslationDataset(test_input, test_output, tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    encoder = Encoder(vocab_size, args.embedding_dim, args.hidden_size, args.num_layers, args.dropout).to(device)
    decoder = Decoder(vocab_size, args.embedding_dim, args.hidden_size, args.num_layers, args.dropout).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = torch.tensor(src).to(device), torch.tensor(tgt).to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({"train_loss": total_loss / len(train_loader)})
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="assignment3_attention")

    # All wandb.config values as arguments
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
