"""
LSTM-based Seq2Seq model for text-to-code generation.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Encoder using LSTM."""
    
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        """
        Args:
            input_size: Size of input vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq, input_lengths=None):
        """
        Args:
            input_seq: Input sequence (batch_size, seq_len)
            input_lengths: Actual lengths of sequences (batch_size)
        
        Returns:
            outputs: All hidden states (batch_size, seq_len, hidden_dim)
            hidden: Tuple of (h_n, c_n) where each is (num_layers, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(input_seq))
        
        if input_lengths is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.lstm(embedded)
        
        return outputs, hidden


class LSTMDecoder(nn.Module):
    """Decoder using LSTM."""
    
    def __init__(self, output_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        """
        Args:
            output_size: Size of output vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden):
        """
        Args:
            input_token: Current input token (batch_size, 1)
            hidden: Previous hidden state - tuple of (h, c) where each is (num_layers, batch_size, hidden_dim)
        
        Returns:
            output: Output logits (batch_size, 1, output_size)
            hidden: New hidden state - tuple of (h, c)
        """
        embedded = self.dropout(self.embedding(input_token))
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden


class LSTMSeq2Seq(nn.Module):
    """Seq2Seq model with LSTM encoder and decoder."""
    
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim=128, 
                 hidden_dim=256, num_layers=1, dropout=0.1):
        """
        Args:
            input_vocab_size: Size of input vocabulary
            output_vocab_size: Size of output vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMSeq2Seq, self).__init__()
        
        self.encoder = LSTMEncoder(input_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(output_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_lengths: Source sequence lengths (batch_size)
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Output logits (batch_size, tgt_len, output_vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.output_size
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Initialize decoder input with SOS token
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        # Decode step by step
        for t in range(1, tgt_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing: use actual target as next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
    
    def generate(self, src, src_lengths, max_length=100, sos_idx=1):
        """
        Generate a sequence using greedy decoding.
        
        Args:
            src: Source sequence (batch_size, src_len)
            src_lengths: Source sequence lengths (batch_size)
            max_length: Maximum generation length
            sos_idx: Start-of-sequence token index
        
        Returns:
            predictions: Generated sequences (batch_size, max_length)
        """
        batch_size = src.size(0)
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Initialize decoder input with SOS token
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long).to(src.device) * sos_idx
        
        # Store predictions
        predictions = torch.zeros(batch_size, max_length, dtype=torch.long).to(src.device)
        predictions[:, 0] = sos_idx
        
        # Generate step by step
        for t in range(1, max_length):
            output, hidden = self.decoder(decoder_input, hidden)
            top1 = output.argmax(2)
            predictions[:, t] = top1.squeeze(1)
            decoder_input = top1
        
        return predictions
