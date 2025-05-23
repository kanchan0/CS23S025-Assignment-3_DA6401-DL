import argparse
import torch
import wandb
from model_utils import *

def main(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    import pandas as pd
    train_data = pd.read_csv(args.train_csv, header=None)
    val_data = pd.read_csv(args.val_csv, header=None)
    test_data = pd.read_csv(args.test_csv, header=None)

    train_input, train_output = train_data[0].to_numpy(), train_data[1].to_numpy()
    val_input, val_output = val_data[0].to_numpy(), val_data[1].to_numpy()
    test_input, test_output = test_data[0].to_numpy(), test_data[1].to_numpy()

    # Remaining code from notebook below:


# Required Imports
import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset, DataLoader
import random
import wandb
torch.manual_seed(42)

wandb.login(key="b4dc866a06ba17317c20de0d13c1a64cc23096dd")

# TO USE GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
train_csv = "/kaggle/input/dakshina-dataset-hindi/DakshinaDataSet_Hindi/hindi_Train_dataset.csv"
test_csv = "/kaggle/input/dakshina-dataset-hindi/DakshinaDataSet_Hindi/hindi_Test_dataset.csv"
val_csv = "/kaggle/input/dakshina-dataset-hindi/DakshinaDataSet_Hindi/hindi_Validation_dataset.csv"

# Data loading
train_data = pd.read_csv(train_csv, header=None)
train_input = train_data[0].to_numpy()
train_output = train_data[1].to_numpy()
val_data = pd.read_csv(val_csv, header=None)
val_input = val_data[0].to_numpy()
val_output = val_data[1].to_numpy()
test_data = pd.read_csv(test_csv, header=None)
test_input = test_data[0].to_numpy()
test_output = test_data[1].to_numpy()


def pre_processing(train_input, train_output):
    data = {
        "all_characters": [],
        "char_num_map": {},
        "num_char_map": {},
        "source_charToNum": torch.zeros(len(train_input), 30, dtype=torch.int, device=device),
        "source_data": train_input,
        "all_characters_2": [],
        "char_num_map_2": {},
        "num_char_map_2": {},
        "val_charToNum": torch.zeros(len(train_output), 23, dtype=torch.int, device=device),
        "target_data": train_output,
        "source_len": 0,
        "target_len": 0
    }
    
    for i in range(len(train_input)):
        train_input[i] = "{" + train_input[i] + "}" * (29 - len(train_input[i]))
        charToNum = []
        for char in train_input[i]:
            if char not in data["all_characters"]:
                data["all_characters"].append(char)
                index = len(data["all_characters"]) - 1
                data["char_num_map"][char] = index
                data["num_char_map"][index] = char
            else:
                index = data["char_num_map"][char]
            charToNum.append(index)
        data["source_charToNum"][i] = torch.tensor(charToNum, device=device)

        train_output[i] = "{" + train_output[i] + "}" * (22 - len(train_output[i]))
        charToNum1 = []
        for char in train_output[i]:
            if char not in data["all_characters_2"]:
                data["all_characters_2"].append(char)
                index = len(data["all_characters_2"]) - 1
                data["char_num_map_2"][char] = index
                data["num_char_map_2"][index] = char
            else:
                index = data["char_num_map_2"][char]
            charToNum1.append(index)
        data["val_charToNum"][i] = torch.tensor(charToNum1, device=device)
    
    data["source_len"] = len(data["all_characters"])
    data["target_len"] = len(data["all_characters_2"])
    return data

data = pre_processing(copy.copy(train_input), copy.copy(train_output))


def pre_processing_validation(val_input, val_output):
    data2 = {
        "source_charToNum": torch.zeros(len(val_input), 30, dtype=torch.int, device=device),
        "val_charToNum": torch.zeros(len(val_output), 23, dtype=torch.int, device=device)
    }
    m1 = data["char_num_map"]
    m2 = data["char_num_map_2"]
    
    for i in range(len(val_input)):
        val_input[i] = "{" + val_input[i] + "}" * (29 - len(val_input[i]))
        charToNum = [m1[char] for char in val_input[i]]
        data2["source_charToNum"][i] = torch.tensor(charToNum, device=device)
        
        val_output[i] = "{" + val_output[i] + "}" * (22 - len(val_output[i]))
        charToNum1 = [m2[char] for char in val_output[i]]
        data2["val_charToNum"][i] = torch.tensor(charToNum1, device=device)
    
    return data2

data2 = pre_processing_validation(copy.copy(val_input), copy.copy(val_output))
data_test = pre_processing_validation(copy.copy(test_input), copy.copy(test_output))

# Custom Dataset class for PyTorch

    def __init__(self, x, y):
        # Store input (x) and target (y) data
        self.source = x
        self.target = y
    

    def __len__(self):
        # Return number of samples
        return len(self.source)
    

    def __getitem__(self, idx):
        # Return a single data sample pair
        return self.source[idx], self.target[idx]

# DataLoader wrapper function for train/validation datasets

def dataLoaderFun(dataName, batch_size):
    if dataName == 'train':
        dataset = MyDataset(data["source_charToNum"], data['val_charToNum'])
    else:
        dataset = MyDataset(data2["source_charToNum"], data2['val_charToNum'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Encoder class using RNN/GRU/LSTM

    def __init__(self, inputDim, embSize, encoderLayers, hiddenLayerNuerons, cellType, bidirection):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(inputDim, embSize)
        self.encoderLayers = encoderLayers
        self.hiddenLayerNuerons = hiddenLayerNuerons
        self.bidirection = bidirection
        self.num_directions = 2 if bidirection == "Yes" else 1

        # Choose RNN cell type
        if cellType == 'GRU':
            self.rnn = nn.GRU(embSize, hiddenLayerNuerons, num_layers=encoderLayers,
                              bidirectional=(bidirection == "Yes"), batch_first=True)
        elif cellType == 'LSTM':
            self.rnn = nn.LSTM(embSize, hiddenLayerNuerons, num_layers=encoderLayers,
                               bidirectional=(bidirection == "Yes"), batch_first=True)
        else:
            self.rnn = nn.RNN(embSize, hiddenLayerNuerons, num_layers=encoderLayers,
                              bidirectional=(bidirection == "Yes"), batch_first=True)
    

    def initHidden(self, batch_size=1):
        h0 = torch.zeros(self.encoderLayers * self.num_directions,
                         batch_size,
                         self.hiddenLayerNuerons,
                         device=device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros_like(h0)
            return (h0, c0)
        else:
            return h0
        

    def forward(self, currentInput, prevState):
        # Embed input and pass through RNN
        embdInput = self.embedding(currentInput)
        return self.rnn(embdInput, prevState)


    def getInitialState(self, batch_size):
        # Create zero initial hidden state
        return torch.zeros(self.encoderLayers * self.num_directions, batch_size, self.hiddenLayerNuerons, device=device)

# Decoder class using RNN/GRU/LSTM

    def __init__(self, outputDim, embSize, hiddenLayerNuerons, decoderLayers, cellType, dropout_p):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(outputDim, embSize)
        self.decoderLayers = decoderLayers

        # Choose RNN cell type
        if cellType == 'GRU':
            self.rnn = nn.GRU(embSize, hiddenLayerNuerons, num_layers=decoderLayers, batch_first=True)
        elif cellType == 'LSTM':
            self.rnn = nn.LSTM(embSize, hiddenLayerNuerons, num_layers=decoderLayers, batch_first=True)
        else:
            self.rnn = nn.RNN(embSize, hiddenLayerNuerons, num_layers=decoderLayers, batch_first=True)

        # Output layer and softmax
        self.fc = nn.Linear(hiddenLayerNuerons, outputDim)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, currentInput, prevState):
        # Forward pass through embedding, RNN, and output layer
        embdInput = self.embedding(currentInput)
        output, prevState = self.rnn(embdInput, prevState)
        output = self.dropout(output)
        output = self.softmax(self.fc(output))
        return output, prevState

# Adjust encoder hidden state for decoder

def init_decoder_state(encoder_state, encoderLayers, decoderLayers, cellType):
    if cellType == 'LSTM':
        h, c = encoder_state
        # Adjust hidden and cell state sizes
        if encoderLayers >= decoderLayers:
            return (h[-decoderLayers:], c[-decoderLayers:])
        else:
            # Duplicate last layer if decoder has more layers
            h_dec = torch.cat([h] + [h[-1:]]*(decoderLayers - encoderLayers), dim=0)
            c_dec = torch.cat([c] + [c[-1:]]*(decoderLayers - encoderLayers), dim=0)
            return (h_dec, c_dec)
    else:
        h = encoder_state
        if encoderLayers >= decoderLayers:
            return h[-decoderLayers:]
        else:
            return torch.cat([h] + [h[-1:]]*(decoderLayers - encoderLayers), dim=0)

# Training loop

def train(embSize, encoderLayers, decoderLayers, hiddenLayerNuerons, cellType, bidirection, dropout, epochs, batchsize, learningRate, optimizer, tf_ratio):
    dataLoader = dataLoaderFun("train", batchsize)
    lossFunction = nn.NLLLoss()
    
    # Initialize encoder and decoder
    encoder = Encoder(data["source_len"], embSize, encoderLayers, hiddenLayerNuerons, cellType, bidirection).to(device)
    decoder = Decoder(data["target_len"], embSize, hiddenLayerNuerons, decoderLayers, cellType, dropout).to(device)

    # Set optimizer
    if optimizer == 'Adam':
        encoderOptimizer = optim.Adam(encoder.parameters(), lr=learningRate)
        decoderOptimizer = optim.Adam(decoder.parameters(), lr=learningRate)
    else:
        encoderOptimizer = optim.NAdam(encoder.parameters(), lr=learningRate)
        decoderOptimizer = optim.NAdam(decoder.parameters(), lr=learningRate)

    for epoch in range(epochs):
        train_accuracy = 0
        train_loss = 0
        total_samples = 0
        
        for batch_num, (sourceBatch, targetBatch) in enumerate(dataLoader):
            current_batch_size = sourceBatch.size(0)
            encoderInitialState = encoder.getInitialState(current_batch_size)

            # Handle bidirection by averaging forward and backward pass input
            if bidirection == "Yes":
                reversed_batch = torch.flip(sourceBatch, dims=[1])
                sourceBatch = (sourceBatch + reversed_batch) // 2

            if cellType == 'LSTM':
                encoderInitialState = (encoderInitialState, torch.zeros_like(encoderInitialState))

            # Forward pass through encoder
            encoder_output, encoderCurrentState = encoder(sourceBatch, encoderInitialState)

            # Reduce bidirectional state
            if bidirection == "Yes":
                if cellType == 'LSTM':
                    encoderCurrentState = (
                        encoderCurrentState[0].view(encoderLayers, 2, current_batch_size, -1).sum(1),
                        encoderCurrentState[1].view(encoderLayers, 2, current_batch_size, -1).sum(1)
                    )
                else:
                    encoderCurrentState = encoderCurrentState.view(encoderLayers, 2, current_batch_size, -1).sum(1)

            decoderCurrState = init_decoder_state(encoderCurrentState, encoderLayers, decoderLayers, cellType)

            loss = 0
            sequenceLen = targetBatch.shape[1]
            Output = []
            randNumber = random.random()

            # Decoder loop
            for i in range(sequenceLen):
                if i == 0:
                    decoderInput = targetBatch[:, i].reshape(current_batch_size, 1)
                else:
                    # Teacher forcing
                    if randNumber < tf_ratio:
                        decoderInput = targetBatch[:, i].reshape(current_batch_size, 1)
                    else:
                        decoderInput = decoderInput.reshape(current_batch_size, 1)

                decoderOutput, decoderCurrState = decoder(decoderInput, decoderCurrState)
                _, topIndeces = decoderOutput.topk(1)
                decoderOutput = decoderOutput[:, -1, :]
                targetChars = targetBatch[:, i].type(dtype=torch.long)
                loss += lossFunction(decoderOutput, targetChars)
                decoderInput = topIndeces.squeeze().detach()
                Output.append(decoderInput)

            # Stack outputs and compute accuracy/loss
            tensor_2d = torch.stack(Output)
            Output = tensor_2d.t()
            train_accuracy += (Output == targetBatch).all(dim=1).sum().item()
            train_loss += (loss.item() / sequenceLen)
            total_samples += targetBatch.size(0)

            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            loss.backward()
            encoderOptimizer.step()
            decoderOptimizer.step()

        # Validation
        val_acc, val_loss = validationAccuracy(encoder, decoder, batchsize, tf_ratio, cellType, bidirection)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(dataLoader),
            "train_accuracy": train_accuracy / total_samples,
            "validation_loss": val_loss / len(dataLoaderFun("validation", batchsize)),
            "validation_accuracy": val_acc / sum(len(b) for b, _ in dataLoaderFun("validation", batchsize))
        })

# Validation function

def validationAccuracy(encoder, decoder, batchsize, tf_ratio, cellType, bidirection):
    dataLoader = dataLoaderFun("validation", batchsize)
    encoder.eval()
    decoder.eval()
    validation_accuracy = 0
    validation_loss = 0
    total_samples = 0
    lossFunction = nn.NLLLoss()

    for batch_num, (sourceBatch, targetBatch) in enumerate(dataLoader):
        current_batch_size = sourceBatch.size(0)
        encoderInitialState = encoder.getInitialState(current_batch_size)

        if cellType == 'LSTM':
            encoderInitialState = (encoderInitialState, torch.zeros_like(encoderInitialState))

        if bidirection == "Yes":
            reversed_batch = torch.flip(sourceBatch, dims=[1])
            sourceBatch = (sourceBatch + reversed_batch) // 2

        encoder_output, encoderCurrentState = encoder(sourceBatch, encoderInitialState)

        if bidirection == "Yes":
            if cellType == 'LSTM':
                encoderCurrentState = (
                    encoderCurrentState[0].view(encoder.encoderLayers, 2, current_batch_size, -1).sum(1),
                    encoderCurrentState[1].view(encoder.encoderLayers, 2, current_batch_size, -1).sum(1)
                )
            else:
                encoderCurrentState = encoderCurrentState.view(encoder.encoderLayers, 2, current_batch_size, -1).sum(1)

        decoderCurrState = init_decoder_state(encoderCurrentState, encoder.encoderLayers, decoder.decoderLayers, cellType)

        loss = 0
        outputSeqLen = targetBatch.shape[1]
        Output = []
        randNumber = random.random()

        for i in range(outputSeqLen):
            if i == 0:
                decoderInputensor = targetBatch[:, i].reshape(current_batch_size, 1)
            else:
                if randNumber < tf_ratio:
                    decoderInputensor = targetBatch[:, i].reshape(current_batch_size, 1)
                else:
                    decoderInputensor = decoderInputensor.reshape(current_batch_size, 1)

            decoderOutput, decoderCurrState = decoder(decoderInputensor, decoderCurrState)
            _, topIndeces = decoderOutput.topk(1)
            decoderOutput = decoderOutput[:, -1, :]
            curr_target_chars = targetBatch[:, i].type(dtype=torch.long)
            loss += lossFunction(decoderOutput, curr_target_chars)
            decoderInputensor = topIndeces.squeeze().detach()
            Output.append(decoderInputensor)

        tensor_2d = torch.stack(Output)
        Output = tensor_2d.t()
        validation_accuracy += (Output == targetBatch).all(dim=1).sum().item()
        validation_loss += (loss.item() / outputSeqLen)
        total_samples += targetBatch.size(0)

    encoder.train()
    decoder.train()
    return validation_accuracy, validation_loss



def main_fun():
    # Initialize a new W&B run for this sweep trial
    wandb.init(project='CS23S025-Assignment-3-DL')

    # Load the current configuration (set by W&B sweep agent)
    params = wandb.config

    # Train the model using the hyperparameters provided by the sweep
    train(
        params.embSize,
        params.encoderLayers,
        params.decoderLayers,
        params.hiddenLayerNuerons,
        params.cellType,
        params.bidirection,
        params.dropout,
        params.epochs,
        params.batchsize,
        params.learningRate,
        params.optimizer,
        params.tf_ratio
    )


sweep_params = {
    'method': 'bayes',  # Use Bayesian optimization (more sample-efficient than grid/random)
    'name': 'Assignment_3_WITHOUT_Attention_2',  # Name of the sweep

    # Define the target metric to optimize
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy',  # Metric reported in wandb.log()
    },

    # Define the hyperparameter search space
    'parameters': {
        'embSize': {'values': [32, 64, 128, 256]},
        'encoderLayers': {'values': [2, 3, 4, 5, 7]},
        'decoderLayers': {'values': [2, 3, 4, 5]},
        'hiddenLayerNuerons': {'values': [64, 256, 512]},
        'cellType': {'values': ['GRU', 'RNN', 'LSTM']},
        'bidirection': {'values': ['no', 'Yes']},  # Note: case-sensitive; use consistently in code
        'dropout': {'values': [0, 0.2, 0.3, 0.5]},
        'epochs': {'values': [10, 15, 20, 25]},
        'batchsize': {'values': [32, 64, 128]},
        'learningRate': {'values': [1e-2, 1e-3, 1e-4]},
        'optimizer': {'values': ['Adam', 'Nadam']},
        'tf_ratio': {'values': [0.2, 0.4, 0.5, 0.7]}  # Teacher forcing ratio
    }
}


sweepId = wandb.sweep(sweep_params, project='CS23S025-Assignment-3-DL')
wandb.agent(sweepId, function=main_fun, count=50, entity="cs23s025-indian-institute-of-technology-madras", project="CS23S025-Assignment-3-DL")

char_to_num_target = data['char_num_map_2']
num_to_char_target = data['num_char_map_2']

print(f"Keys in the 'data' dictionary: {data.keys()}")

import torch
from torch.utils.data import DataLoader
import editdistance
import csv

# Character-to-index and index-to-character mappings for both source and target sequences
char_to_num_target = data['char_num_map_2']
num_to_char_target = data['num_char_map_2']
char_to_num_source = data['char_num_map']
num_to_char_source = data['num_char_map']

# Hyperparameters for model architecture (must match the training config)
embSize = 32
encoderLayers = 3
decoderLayers = 3
hiddenLayerNuerons = 512
cellType = "GRU"
bidirection = 'no'
dropout = 0.3
epochs = 15
batchsize = 64
learningRate = 0.001
optimizer = 'Nadam'
tf_ratio = 1.0

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize encoder and decoder models with the same architecture as during training
encoder = Encoder(
    inputDim=data["source_len"],
    embSize=embSize,
    encoderLayers=encoderLayers,
    hiddenLayerNuerons=hiddenLayerNuerons,
    cellType=cellType,
    bidirection=bidirection
).to(device)

decoder = Decoder(
    outputDim=data["target_len"],
    embSize=embSize,
    hiddenLayerNuerons=hiddenLayerNuerons,
    decoderLayers=decoderLayers,
    cellType=cellType,
    dropout_p=dropout
).to(device)

# Load the best saved model weights from training
checkpoint = torch.load("best_model.pth", map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder.eval()
decoder.eval()

# Prepare the test dataset and data loader (batch_size=1 for evaluation)
test_dataset = MyDataset(data_test["source_charToNum"], data_test["val_charToNum"])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Maximum length of the target sequence for inference
max_target_len = 23

# Counters to compute test accuracy
correct = 0
total = 0
print_limit = 10  # Limit the number of verbose printed examples

# Utility function to clean predicted/target sequences by removing special tokens

def clean(seq):
    return ''.join(c for c in seq if c not in ('{', '}'))

# Evaluation loop (inference without teacher forcing)
with torch.no_grad():
    for source_tensor, target_tensor in test_loader:
        source_tensor = source_tensor.to(device)
        target_tensor = target_tensor.to(device)

        # Initialize encoder hidden state
        enc_hidden = encoder.initHidden(batch_size=1)

        # Pass input through the encoder
        encoder_outputs, enc_hidden = encoder(source_tensor, enc_hidden)

        # Initialize decoder hidden state using encoder's final state
        dec_hidden = init_decoder_state(enc_hidden, encoderLayers, decoderLayers, cellType)

        # Start token for the decoder
        decoder_input = torch.tensor([[char_to_num_target["{"]]], device=device)

        decoded_output = []  # Store predicted characters

        # Generate one character at a time until max length or end token is reached
        for _ in range(max_target_len):
            decoder_output, dec_hidden = decoder(decoder_input, dec_hidden)
            topv, topi = decoder_output.topk(1)
            next_index = topi.item()  # Get index of predicted token
            next_char = num_to_char_target[next_index]

            if next_char == "}":
                break  # Stop decoding at end token

            decoded_output.append(next_char)
            decoder_input = torch.tensor([[next_index]], device=device)

        # Convert input, target, and predicted sequences from index to character
        input_seq = clean([num_to_char_source[i.item()] for i in source_tensor[0]])
        target_seq = clean([num_to_char_target[i.item()] for i in target_tensor[0]])
        predicted_seq = clean(decoded_output)

        # Compare prediction to ground truth
        if predicted_seq == target_seq:
            correct += 1
        total += 1

        # Print the first few predictions and some correct/incorrect cases
        if total <= print_limit:
            print(f"Input:     {input_seq}")
            print(f"Target:    {target_seq}")
            print(f"Predicted: {predicted_seq}\n")

        if total <= 3 or (total <= 20 and predicted_seq == target_seq):
            print(f"MATCH! Input: {input_seq} | Target: {target_seq} | Predicted: {predicted_seq}")
        elif total <= 20:
            print(f"DIFF!  Input: {input_seq} | Target: {target_seq} | Predicted: {predicted_seq}")

# Compute and print final test accuracy
accuracy = correct / total * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    # Add more arguments here as needed (e.g., epochs, learning rate)
    args = parser.parse_args()
    main(args)
