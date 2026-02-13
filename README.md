# Text-to-Python Code Generation Using Seq2Seq Models

This project implements and compares three different recurrent neural network (RNN) architectures for text-to-code generation, where natural language function descriptions (docstrings) are translated into Python source code.

## Models Implemented

1. **Vanilla RNN-based Seq2Seq**: Basic sequence-to-sequence model using vanilla RNN cells
2. **LSTM-based Seq2Seq**: Improved sequence-to-sequence model using LSTM cells
3. **LSTM with Attention**: Advanced model using LSTM with attention mechanism for better long-range dependencies

## Project Structure

```
.
├── README.md                   # This file
├── MODEL_ARCHITECTURE.md       # Detailed model architecture documentation
├── requirements.txt            # Python dependencies
├── quick_start.py             # Quick start example script
├── data_utils.py              # Data preprocessing and dataset utilities
├── model_rnn.py               # Vanilla RNN Seq2Seq implementation
├── model_lstm.py              # LSTM Seq2Seq implementation
├── model_lstm_attention.py    # LSTM with Attention implementation
├── train_utils.py             # Training and evaluation utilities
├── train.py                   # Main training script
└── checkpoints/               # Saved model checkpoints (created during training)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/faria1305/Text-to-Python-Code-Generation-Using-Seq2Seq-Models.git
cd Text-to-Python-Code-Generation-Using-Seq2Seq-Models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

The fastest way to get started:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the quick start example
python quick_start.py
```

This will train all three models and generate a comparison plot.

## Detailed Usage

### Training All Models

Train and compare all three models:
```bash
python train.py --model all --num_epochs 50 --batch_size 4
```

### Training Individual Models

Train a specific model:
```bash
# Vanilla RNN
python train.py --model rnn --num_epochs 50

# LSTM
python train.py --model lstm --num_epochs 50

# LSTM with Attention
python train.py --model attention --num_epochs 50
```

### Command-line Arguments

- `--model`: Model to train (choices: 'rnn', 'lstm', 'attention', 'all'; default: 'all')
- `--embedding_dim`: Dimension of embeddings (default: 128)
- `--hidden_dim`: Dimension of hidden state (default: 256)
- `--num_layers`: Number of RNN/LSTM layers (default: 1)
- `--dropout`: Dropout probability (default: 0.1)
- `--batch_size`: Batch size (default: 4)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--max_text_len`: Maximum text sequence length (default: 50)
- `--max_code_len`: Maximum code sequence length (default: 100)
- `--save_dir`: Directory to save model checkpoints (default: 'checkpoints')
- `--device`: Device to train on (default: 'cuda' if available, else 'cpu')

## Example Output

When training all models, you'll see:
- Training progress for each model
- Training and validation losses per epoch
- Sample predictions comparing target code vs. generated code
- A comparison plot showing training and validation losses across all models

## Dataset

The project includes a sample dataset of docstring-code pairs for demonstration purposes. The dataset includes simple Python functions like:
- Mathematical operations (add, multiply, subtract, etc.)
- String operations (uppercase, lowercase, reverse, etc.)
- List operations (first, last, append, etc.)
- Utility functions (absolute value, square, cube, etc.)

For real-world applications, you can extend the `create_sample_dataset()` function in `data_utils.py` to load larger datasets.

## Model Architectures

### 1. Vanilla RNN Seq2Seq
- **Encoder**: Vanilla RNN that processes the input text sequence
- **Decoder**: Vanilla RNN that generates the output code sequence
- Uses teacher forcing during training for faster convergence
- Simple architecture, good baseline model

### 2. LSTM Seq2Seq
- **Encoder**: LSTM network that processes the input text sequence
- **Decoder**: LSTM network that generates the output code sequence
- Better at capturing long-range dependencies compared to vanilla RNN
- Uses memory cells to maintain information over time

### 3. LSTM with Attention
- **Encoder**: LSTM network that processes the input text sequence
- **Decoder**: LSTM network with attention mechanism
- **Attention**: Computes context vector as weighted sum of encoder outputs
- Allows the decoder to focus on relevant parts of the input sequence
- Best performance for sequence-to-sequence tasks

For detailed architecture information, see [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

## Performance Comparison

The models are evaluated based on:
- **Training Loss**: Cross-entropy loss on training data
- **Validation Loss**: Cross-entropy loss on validation data
- **Sample Predictions**: Qualitative evaluation of generated code

When training all models together, a comparison plot is automatically generated showing the training and validation losses for each model.

## Future Improvements

Potential enhancements for this project:
- Add beam search for better decoding
- Implement more sophisticated attention mechanisms (e.g., multi-head attention)
- Use transformer-based architectures
- Train on larger, real-world datasets (e.g., CodeSearchNet)
- Add BLEU score and other quantitative metrics
- Implement model ensembling

## License

This project is open source and available under the MIT License.

## Contributors

- faria1305

## Acknowledgments

This project was developed as part of an exploration into different RNN architectures for sequence-to-sequence learning tasks.