# Seq2Seq Text-to-Python Code Generation (PyTorch)

Implements and compares:
1) Vanilla RNN Seq2Seq
2) LSTM Seq2Seq
3) BiLSTM encoder + Bahdanau Attention + LSTM decoder

Dataset: CodeSearchNet Python (`Nan-Do/code-search-net-python`).

## One-command (Docker)
```bash
docker build -t seq2seq-text2code .
docker run --rm -it -v "$PWD/outputs:/app/outputs" seq2seq-text2code
```

## One-command (Local)
```bash
pip install -r requirements.txt
bash run.sh all
```

## Outputs
- `outputs/checkpoints/` model checkpoints
- `outputs/logs/` JSON metrics + decoded samples
- `outputs/plots/` loss curves + docstring-length analysis
- `outputs/attention/` attention heatmaps (attention model)

### Note on splits (IMPORTANT)
The assignment dataset `Nan-Do/code-search-net-python` exposes a single HF split (`train`) but includes a `partition` column with labels `train/valid/test`.  
This project **automatically builds** train/validation/test splits from that column to match the assignment requirement of using the same split for all models.
