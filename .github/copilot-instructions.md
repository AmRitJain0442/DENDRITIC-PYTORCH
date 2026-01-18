# Giant-Killer NLP Project - Copilot Instructions

This is a PyTorch-based NLP project for the Dendritic Optimization Hackathon. The project uses Perforated Backpropagation to enhance a BERT-Tiny model (4M parameters) to achieve BERT-Base level performance (110M parameters) for toxicity classification.

## Project Overview

- **Framework**: PyTorch with Transformers and PerforatedAI
- **Task**: Toxicity Classification using Jigsaw Dataset
- **Architecture**: BERT-Tiny with Dendritic Optimization
- **Goal**: Achieve <2% performance gap vs BERT-Base with 15-40x speed improvement

## Key Components

1. **Data Module** (`src/data/`): Jigsaw dataset loading and preprocessing
2. **Model Module** (`src/models/`): BERT-Tiny with dendritic wrapping
3. **Training** (`src/training/`): Perforated training loop with PAI tracker
4. **Evaluation** (`src/evaluation/`): Benchmarking and comparison scripts

## Development Guidelines

- Use `max_length=128` for tokenization to ensure fast inference
- Do not call `scheduler.step()` manually - PAI tracker handles scheduling
- Always wrap model with `UPA.initialize_pai(model)` before training
- Use `torch.quantization.quantize_dynamic` for edge deployment

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Evaluate and benchmark
python src/evaluate.py
```
