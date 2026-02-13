"""
Training utilities for Seq2Seq models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        text = batch['text'].to(device)
        code = batch['code'].to(device)
        text_len = batch['text_len'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(text, code, text_len, teacher_forcing_ratio)
        
        # Reshape for loss calculation
        outputs = outputs[:, 1:].reshape(-1, outputs.size(-1))
        targets = code[:, 1:].reshape(-1)
        
        # Calculate loss (ignore padding)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text = batch['text'].to(device)
            code = batch['code'].to(device)
            text_len = batch['text_len'].to(device)
            
            # Forward pass without teacher forcing
            outputs = model(text, code, text_len, teacher_forcing_ratio=0.0)
            
            # Reshape for loss calculation
            outputs = outputs[:, 1:].reshape(-1, outputs.size(-1))
            targets = code[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def calculate_bleu_score(predictions, targets, vocab):
    """
    Calculate a simple approximation of BLEU score.
    For a more accurate BLEU, use nltk or sacrebleu libraries.
    """
    from collections import Counter
    
    total_score = 0
    count = 0
    
    for pred, tgt in zip(predictions, targets):
        # Convert to tokens
        pred_tokens = [vocab.get_token(idx.item()) for idx in pred]
        tgt_tokens = [vocab.get_token(idx.item()) for idx in tgt]
        
        # Remove special tokens
        pred_tokens = [t for t in pred_tokens if t not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN, vocab.EOS_TOKEN]]
        tgt_tokens = [t for t in tgt_tokens if t not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN, vocab.EOS_TOKEN]]
        
        if len(tgt_tokens) == 0:
            continue
        
        # Simple unigram precision
        pred_counter = Counter(pred_tokens)
        tgt_counter = Counter(tgt_tokens)
        
        overlap = sum((pred_counter & tgt_counter).values())
        precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0
        
        total_score += precision
        count += 1
    
    return total_score / count if count > 0 else 0


def calculate_accuracy(predictions, targets, vocab):
    """Calculate token-level accuracy."""
    total_correct = 0
    total_tokens = 0
    
    for pred, tgt in zip(predictions, targets):
        # Ignore padding tokens
        mask = tgt != vocab.token2idx[vocab.PAD_TOKEN]
        correct = (pred == tgt) & mask
        
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
    
    return total_correct / total_tokens if total_tokens > 0 else 0


def train_model(model, train_loader, val_loader, num_epochs, device, 
                learning_rate=0.001, model_name='model', save_dir='checkpoints'):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        device: Device to train on
        learning_rate: Learning rate for optimizer
        model_name: Name for saving the model
        save_dir: Directory to save checkpoints
    
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss function (ignore padding index)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio=0.5)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, f'{model_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')
    
    return train_losses, val_losses


def generate_predictions(model, dataloader, device, vocab, num_samples=5):
    """
    Generate predictions and compare with targets.
    
    Args:
        model: The trained model
        dataloader: Data loader
        device: Device to run on
        vocab: Code vocabulary for decoding
        num_samples: Number of samples to generate
    
    Returns:
        results: List of dictionaries with source, target, and prediction
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            text = batch['text'].to(device)
            code = batch['code'].to(device)
            text_len = batch['text_len'].to(device)
            
            # Generate predictions
            predictions = model.generate(text, text_len, max_length=code.size(1), sos_idx=vocab.token2idx[vocab.SOS_TOKEN])
            
            # Convert to tokens
            for i in range(min(num_samples, text.size(0))):
                pred_tokens = [vocab.get_token(idx.item()) for idx in predictions[i]]
                tgt_tokens = [vocab.get_token(idx.item()) for idx in code[i]]
                
                # Remove special tokens and padding
                pred_tokens = [t for t in pred_tokens if t not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN]]
                tgt_tokens = [t for t in tgt_tokens if t not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN]]
                
                # Stop at EOS
                if vocab.EOS_TOKEN in pred_tokens:
                    pred_tokens = pred_tokens[:pred_tokens.index(vocab.EOS_TOKEN)]
                if vocab.EOS_TOKEN in tgt_tokens:
                    tgt_tokens = tgt_tokens[:tgt_tokens.index(vocab.EOS_TOKEN)]
                
                results.append({
                    'prediction': ' '.join(pred_tokens),
                    'target': ' '.join(tgt_tokens),
                })
                
                if len(results) >= num_samples:
                    return results
    
    return results
