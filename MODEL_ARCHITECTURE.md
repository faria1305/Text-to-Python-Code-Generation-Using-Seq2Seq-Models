# Model Architecture Documentation

This document provides detailed information about the three Seq2Seq model architectures implemented in this project.

## Overview

All three models follow the encoder-decoder architecture:
1. **Encoder**: Processes the input sequence (natural language description) and creates a context representation
2. **Decoder**: Generates the output sequence (Python code) based on the encoder's context

The key differences lie in the type of recurrent cells used and whether attention mechanism is employed.

## 1. Vanilla RNN-based Seq2Seq

### Architecture
```
Input Text → Embedding → RNN Encoder → Context Vector → RNN Decoder → Output Code
```

### Components

#### Encoder (RNNEncoder)
- **Input**: Tokenized text sequence
- **Embedding Layer**: Maps tokens to dense vectors
- **RNN Layers**: Processes sequence sequentially
- **Output**: Final hidden state (context vector) and all hidden states

#### Decoder (RNNDecoder)
- **Input**: Previous token and hidden state
- **Embedding Layer**: Maps tokens to dense vectors
- **RNN Layers**: Generates next token based on previous state
- **Linear Layer**: Projects hidden state to vocabulary size
- **Output**: Token probabilities

### Key Features
- Simple architecture, easy to understand
- Fast training compared to LSTM
- Limited ability to capture long-range dependencies
- Suffers from vanishing gradient problem on longer sequences

### Use Cases
- Good baseline model
- Suitable for short sequences
- Educational purposes to understand Seq2Seq basics

---

## 2. LSTM-based Seq2Seq

### Architecture
```
Input Text → Embedding → LSTM Encoder → Context Vector → LSTM Decoder → Output Code
```

### Components

#### Encoder (LSTMEncoder)
- **Input**: Tokenized text sequence
- **Embedding Layer**: Maps tokens to dense vectors
- **LSTM Layers**: Processes sequence with memory cells
- **Output**: Final hidden state (h) and cell state (c), plus all hidden states

#### Decoder (LSTMDecoder)
- **Input**: Previous token and hidden/cell states
- **Embedding Layer**: Maps tokens to dense vectors
- **LSTM Layers**: Generates next token with long-term memory
- **Linear Layer**: Projects hidden state to vocabulary size
- **Output**: Token probabilities

### Key Features
- Better at capturing long-range dependencies than vanilla RNN
- Uses memory cells (cell state) to maintain information over time
- Less prone to vanishing gradient problem
- More parameters than vanilla RNN (slower training)

### LSTM Cell Operations
The LSTM uses gates to control information flow:
- **Forget Gate**: Decides what information to discard from cell state
- **Input Gate**: Decides what new information to add to cell state
- **Output Gate**: Decides what information to output

### Use Cases
- Standard choice for sequence-to-sequence tasks
- Good for medium to long sequences
- Better generalization than vanilla RNN

---

## 3. LSTM with Attention Mechanism

### Architecture
```
Input Text → Embedding → LSTM Encoder → All Hidden States
                                              ↓
                                      Attention Mechanism
                                              ↓
Previous Token → Embedding → Concat with Context → LSTM Decoder → Output Code
```

### Components

#### Encoder (AttentionLSTMEncoder)
- Same as LSTM encoder
- **Crucially**: Returns ALL hidden states, not just the final one

#### Attention Mechanism
- **Input**: Decoder's previous hidden state and all encoder hidden states
- **Process**:
  1. Compute attention scores between decoder state and each encoder state
  2. Apply softmax to get attention weights
  3. Compute weighted sum of encoder states (context vector)
- **Output**: Context vector and attention weights

#### Decoder (AttentionLSTMDecoder)
- **Input**: Previous token, hidden/cell states, and all encoder outputs
- **Embedding Layer**: Maps tokens to dense vectors
- **Attention Layer**: Computes context vector
- **LSTM Layers**: Takes concatenation of embedding and context
- **Linear Layer**: Projects concatenation of LSTM output and context to vocabulary
- **Output**: Token probabilities and attention weights

### Key Features
- Can focus on different parts of input for each output token
- Much better at handling long sequences
- Interpretable (can visualize attention weights)
- More parameters and slower training than basic LSTM
- State-of-the-art performance for sequence-to-sequence tasks

### Attention Score Calculation
```
score(h_t, h_s) = v^T * tanh(W * [h_t; h_s])
attention_weights = softmax(scores)
context = Σ(attention_weights * encoder_outputs)
```

Where:
- `h_t` = current decoder hidden state
- `h_s` = encoder hidden states
- `W`, `v` = learnable parameters

### Use Cases
- Best choice for text-to-code generation
- Essential for long input sequences
- When interpretability is important
- State-of-the-art sequence-to-sequence tasks

---

## Model Comparison

| Feature | Vanilla RNN | LSTM | LSTM + Attention |
|---------|-------------|------|------------------|
| Parameters | ~227K | ~820K | ~1.2M |
| Training Speed | Fastest | Medium | Slowest |
| Long-range Dependencies | Poor | Good | Excellent |
| Gradient Issues | High | Low | Low |
| Interpretability | Low | Low | High (via attention) |
| Best For | Short sequences | Medium sequences | Long sequences |

## Training Strategy

All models use:
- **Loss Function**: Cross-entropy loss (ignoring padding)
- **Optimizer**: Adam
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Teacher Forcing**: 50% probability during training
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients

### Teacher Forcing
During training, with 50% probability:
- **With teacher forcing**: Use actual target token as next input
- **Without teacher forcing**: Use model's prediction as next input

This helps the model learn faster while still being robust to its own errors.

## Generation Strategy

During inference (generation), all models use:
- **Greedy Decoding**: Select the most probable token at each step
- **No Teacher Forcing**: Always use model's own predictions
- **Stop Conditions**: Generate until EOS token or max length reached

## Future Improvements

Potential enhancements:
1. **Beam Search**: Instead of greedy decoding, maintain top-k hypotheses
2. **Bidirectional Encoder**: Process input in both directions
3. **Multi-head Attention**: Multiple attention mechanisms in parallel
4. **Transformer Architecture**: Replace RNN/LSTM with self-attention
5. **Copy Mechanism**: Allow copying tokens directly from input
6. **Coverage Mechanism**: Prevent attending to same positions repeatedly

## References

1. Sutskever et al. (2014) - "Sequence to Sequence Learning with Neural Networks"
2. Bahdanau et al. (2015) - "Neural Machine Translation by Jointly Learning to Align and Translate"
3. Luong et al. (2015) - "Effective Approaches to Attention-based Neural Machine Translation"
