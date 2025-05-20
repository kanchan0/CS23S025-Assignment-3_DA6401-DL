# DA6401-Assignment-3 (CS23S025)
# Sequence-to-Sequence Learning with RNNs

## Overview

This project explores sequence-to-sequence (seq2seq) learning using Recurrent Neural Networks (RNNs) and their advanced variants. It aims to provide practical insights into the architecture, training process, and limitations of vanilla RNNs while showcasing improvements through LSTM, GRU, and attention mechanisms.

## Objectives

- Model sequence-to-sequence problems using different RNN-based architectures.
- Compare the behavior and performance of Vanilla RNN, LSTM, and GRU cells.
- Understand and implement attention mechanisms to address limitations of basic seq2seq models.
- Visualize how various components interact in an RNN-based model during training and inference.


## Project Structure


### File Descriptions

- **`cs23s025_Assignment3_Without_Attention.ipynb`**  
  This notebook implements a standard sequence-to-sequence (seq2seq) learning model using Recurrent Neural Networks (RNNs) without any attention mechanism. It includes comparisons among Vanilla RNN, LSTM, and GRU cells and evaluates their performance on a given task.

- **`cs23s025_Assignment3_With_Attention.ipynb`**  
  This notebook extends the previous model by incorporating an attention mechanism. It demonstrates how attention enhances the model's ability to handle long sequences and improves prediction accuracy. Visualizations of attention weights are included to aid interpretation.

All relevant parameters, such as model architecture choices, training settings, and evaluation metrics, are clearly described within each notebook.


##  Requirements

This project requires the following Python packages:

- **torch**: Core deep learning framework used to build and train RNN-based models.
- **pandas**: Used for any tabular data handling or preprocessing.
- **wandb**: (Weights & Biases) Used for tracking experiments, logging metrics, and visualizing training performance.

To install all dependencies, run:

```bash
pip install -r requirements.txt

```


## Instructions to Run

You can run this project either on **Kaggle** or on your **local machine**.

###  On Kaggle (Recommended)

1. Open the `.ipynb` notebook of your choice:
   - `cs23s025_Assignment3_Without_Attention.ipynb`
   - `cs23s025_Assignment3_With_Attention.ipynb`

2. Ensure the **GPU** accelerator is enabled:
   - Go to *Settings * (top-left)
   - Turn on **GPU**

3. Run all cells sequentially:
   - Click **"Run All"** or execute each cell using `Shift + Enter`.

No extra installation is required — all packages are preinstalled in the Kaggle environment.

---
### On Local Machine

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/kanchan0/CS23S025-Assignment-3_DA6401-DL.git
   cd your-repo-name
   pip install -r requirements.txt 

2. Open the `.ipynb` notebook of your choice:
   - `cs23s025_Assignment3_Without_Attention.ipynb`
   - `cs23s025_Assignment3_With_Attention.ipynb`

3. Run all cells sequentially:
   - Click **"Run All"** or execute each cell using `Shift + Enter`. 

The following table summarizes the hyperparameters used in the model, with default values for both **with** and **without** attention mechanisms.

|         Name          | Default_Values-Without Attention | Default_Values-With Attention | Description                                                      |
|:---------------------:|:-----------------:|:--------------:|:-----------------------------------------------------------------|
|    `embSize`          |        64         |       64       | Embedding size dimension used in the encoder and decoder         |
|  `encoderLayers`      |         1         |        1       | Number of layers in the encoder                                  |
|  `decoderLayers`      |         1         |        1       | Number of layers in the decoder                                  |
| `hiddenLayerNuerons`  |       512         |      512       | Number of neurons in each hidden layer of encoder/decoder        |
|     `cellType`        |      `'LSTM'`      |    `'GRU'`     | Type of RNN cell used: choices are `GRU`, `RNN`, or `LSTM`       |
|    `bidirection`      |      `'no'`       |    `'no'`      | Use bidirectional RNN encoder? Options: `'no'` or `'Yes'`        |
|      `dropout`        |       0.2         |      0.2       | Dropout rate applied to RNN layers                               |
|       `epochs`        |        15         |       15       | Number of training epochs                                        |
|     `batchsize`       |        64         |       64       | Batch size during training                                       |
|    `learningRate`     |      0.001        |     0.001      | Learning rate for optimizer                                      |
|     `optimizer`       |     `'NAdam'`      |   `'Adam'`     | Optimizer choice: `'Adam'` or `'Nadam'`                         |
|      `tf_ratio`       |       0.5         |      0.5       | Teacher forcing ratio during decoder training                    |


---

### File paths used in the code (can be modified as needed):

|      Variable       |                                   Path                                   | Description                      |
| :-----------------: | :----------------------------------------------------------------------: | :-------------------------------|
|    `train_csv`      | `/kaggle/input/dakshina-dataset-hindi/DakshinaDataSet_Hindi/hindi_Train_dataset.csv` | Path to training dataset CSV     |
|     `test_csv`      | `/kaggle/input/dakshina-dataset-hindi/DakshinaDataSet_Hindi/hindi_Test_dataset.csv`  | Path to test dataset CSV         |
|     `val_csv`       | `/kaggle/input/dakshina-dataset-hindi/DakshinaDataSet_Hindi/hindi_Validation_dataset.csv` | Path to validation dataset CSV   |
