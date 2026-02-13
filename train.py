"""
Main script for training and comparing Seq2Seq models.

This script implements and compares three different RNN architectures:
1. Vanilla RNN-based Seq2Seq
2. LSTM-based Seq2Seq
3. LSTM with Attention mechanism
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import matplotlib.pyplot as plt

from data_utils import create_sample_dataset, build_vocabulary, CodeDataset
from model_rnn import VanillaRNNSeq2Seq
from model_lstm import LSTMSeq2Seq
from model_lstm_attention import LSTMAttentionSeq2Seq
from train_utils import train_model, generate_predictions


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train and compare Seq2Seq models for text-to-code generation')
    
    parser.add_argument('--model', type=str, default='all', choices=['rnn', 'lstm', 'attention', 'all'],
                        help='Model to train: rnn, lstm, attention, or all (default: all)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of embeddings (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Dimension of hidden state (default: 256)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of RNN/LSTM layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--max_text_len', type=int, default=50,
                        help='Maximum text sequence length (default: 50)')
    parser.add_argument('--max_code_len', type=int, default=100,
                        help='Maximum code sequence length (default: 100)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models (default: checkpoints)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (default: cuda if available, else cpu)')
    
    return parser.parse_args()


def prepare_data(max_text_len, max_code_len, batch_size):
    """Prepare training and validation datasets."""
    print("Preparing data...")
    
    # Create sample dataset
    texts, codes = create_sample_dataset()
    
    # Build vocabularies
    text_vocab = build_vocabulary(texts, min_freq=1)
    code_vocab = build_vocabulary(codes, min_freq=1)
    
    print(f"Text vocabulary size: {len(text_vocab)}")
    print(f"Code vocabulary size: {len(code_vocab)}")
    
    # Create dataset
    dataset = CodeDataset(texts, codes, text_vocab, code_vocab, max_text_len, max_code_len)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    return train_loader, val_loader, text_vocab, code_vocab


def create_model(model_type, input_vocab_size, output_vocab_size, embedding_dim, 
                 hidden_dim, num_layers, dropout, device):
    """Create the specified model."""
    if model_type == 'rnn':
        model = VanillaRNNSeq2Seq(
            input_vocab_size, output_vocab_size, embedding_dim, 
            hidden_dim, num_layers, dropout
        )
    elif model_type == 'lstm':
        model = LSTMSeq2Seq(
            input_vocab_size, output_vocab_size, embedding_dim, 
            hidden_dim, num_layers, dropout
        )
    elif model_type == 'attention':
        model = LSTMAttentionSeq2Seq(
            input_vocab_size, output_vocab_size, embedding_dim, 
            hidden_dim, num_layers, dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def plot_losses(results, save_path='training_comparison.png'):
    """Plot training and validation losses for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name, (train_losses, val_losses) in results.items():
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, label=model_name, marker='o')
        ax2.plot(epochs, val_losses, label=model_name, marker='s')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining comparison plot saved to {save_path}")
    plt.close()


def main():
    """Main training function."""
    args = parse_args()
    
    print("="*60)
    print("Text-to-Python Code Generation using Seq2Seq Models")
    print("="*60)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Prepare data
    train_loader, val_loader, text_vocab, code_vocab = prepare_data(
        args.max_text_len, args.max_code_len, args.batch_size
    )
    
    # Determine which models to train
    if args.model == 'all':
        model_types = ['rnn', 'lstm', 'attention']
    else:
        model_types = [args.model]
    
    # Train models
    results = {}
    
    for model_type in model_types:
        print("\n" + "="*60)
        print(f"Training {model_type.upper()} Model")
        print("="*60)
        
        # Create model
        model = create_model(
            model_type,
            len(text_vocab),
            len(code_vocab),
            args.embedding_dim,
            args.hidden_dim,
            args.num_layers,
            args.dropout,
            args.device
        )
        
        print(f"\nModel architecture:")
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            args.num_epochs,
            args.device,
            args.learning_rate,
            model_name=model_type,
            save_dir=args.save_dir
        )
        
        results[model_type] = (train_losses, val_losses)
        
        # Generate sample predictions
        print("\n" + "-"*60)
        print("Sample Predictions:")
        print("-"*60)
        
        predictions = generate_predictions(model, val_loader, args.device, code_vocab, num_samples=5)
        for i, pred in enumerate(predictions, 1):
            print(f"\nSample {i}:")
            print(f"  Target:     {pred['target']}")
            print(f"  Prediction: {pred['prediction']}")
    
    # Plot comparison if multiple models were trained
    if len(model_types) > 1:
        plot_losses(results)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Print final results summary
    print("\nFinal Results:")
    print("-"*60)
    for model_type in model_types:
        train_losses, val_losses = results[model_type]
        print(f"{model_type.upper():10s} - Final Train Loss: {train_losses[-1]:.4f}, Final Val Loss: {val_losses[-1]:.4f}")


if __name__ == '__main__':
    main()
