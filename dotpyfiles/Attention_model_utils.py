import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = self.tokenizer.encode(self.inputs[idx])
        output_tensor = self.tokenizer.encode(self.outputs[idx])
        return input_tensor, output_tensor

# Example tokenizer placeholder
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.reverse_vocab = {0: "<pad>", 1: "<sos>", 2: "<eos>"}

    def encode(self, text):
        return [1] + [ord(c) % 256 for c in text] + [2]  # simple char encoding

    def decode(self, tokens):
        return ''.join([chr(t) for t in tokens if t > 2])

# Attention Module
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden[-1].unsqueeze(1).repeat(1, timestep, 1)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention, dim=1)

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers,
                          dropout=dropout, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, hidden = self.rnn(x)
        return outputs, hidden

# Decoder with Attention
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_size, hidden_size, num_layers,
                          dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        x = self.embedding(x)
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((x, context), dim=2)
        outputs, hidden = self.rnn(rnn_input, hidden)
        predictions = self.fc(outputs)
        return predictions, hidden

# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_outputs, hidden = self.encoder(src)
        output, _ = self.decoder(tgt, hidden, encoder_outputs)
        return output
