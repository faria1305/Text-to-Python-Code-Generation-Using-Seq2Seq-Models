"""
Quick start example for training and comparing Seq2Seq models.

This script demonstrates how to quickly train and compare all three models
with minimal configuration.
"""

import subprocess
import sys

def main():
    """Run a quick training example."""
    print("=" * 70)
    print("Quick Start Example: Text-to-Python Code Generation")
    print("=" * 70)
    print("\nThis example will train all three models (RNN, LSTM, LSTM+Attention)")
    print("for 10 epochs each and generate a comparison plot.")
    print("\nTraining parameters:")
    print("  - Models: All (RNN, LSTM, LSTM+Attention)")
    print("  - Epochs: 10")
    print("  - Batch size: 4")
    print("  - Learning rate: 0.001")
    print("  - Device: CPU (use --device cuda if you have GPU)")
    print("\nStarting training...")
    print("-" * 70)
    
    # Run training
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "train.py",
        "--model", "all",
        "--num_epochs", "10",
        "--batch_size", "4",
        "--learning_rate", "0.001"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)
        print("\nResults:")
        print("  - Model checkpoints saved in 'checkpoints/' directory")
        print("  - Training comparison plot saved as 'training_comparison.png'")
        print("\nNext steps:")
        print("  1. View the comparison plot to see model performance")
        print("  2. Check sample predictions in the output above")
        print("  3. Try training individual models with different hyperparameters")
        print("\nFor more options, run: python train.py --help")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
