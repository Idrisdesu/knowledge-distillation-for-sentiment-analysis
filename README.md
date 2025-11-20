# Optimization of LLMs for Real-Time Sentiment Analysis

This project explores the compression of large language models (LLMs) for real-time sentiment analysis tasks. The primary goal is to distill the knowledge from a large, fine-tuned `roberta-large` model (the "Teacher") into smaller, faster "student" models (like DistilBERT, MiniLM, and TinyBERT) and to further optimize them using post-training quantization (PTQ).

The experiments are conducted on two distinct datasets:
- **IMDb**: A large dataset for binary sentiment classification (positive/negative).
- **TweetEval (Sentiment)**: A more complex, multi-class dataset (positive/negative/neutral) with shorter texts, requiring more nuanced understanding.

## Key Results & Insights

The distillation process was highly effective, producing lightweight models that retain a significant portion of the teacher's performance while being substantially faster.

| Model | Accuracy (IMDb) | Accuracy (TweetEval) | Speedup (vs. Teacher) |
|---|---|---|---|
| **Teacher (roberta-large)** | ~95% | ~89% | 1x |
| **DistilBERT** | >92% | >85% | ~8x |
| **MiniLM** | >91% | >84% | ~6x |
| **TinyBERT** | >90% | >82% | **~12x** |
| **TinyBERT (INT8 Quantized)** | >90% | >82% | **~15x** |

*Note: These are representative results. Actual numbers can be found in the `results/` directory.*

### The Importance of Teacher Calibration

A key insight from this project is that the *quality* of the teacher's predictions is more important than its raw accuracy.

- On the **IMDb dataset**, the RoBERTa-large teacher achieved very high accuracy (~95%), leading to "arrogant" or over-confident predictions (e.g., probabilities of 0.999 for one class).
- On the more complex **TweetEval dataset**, the teacher was less certain, producing more nuanced or "calibrated" predictions (e.g., probabilities of 0.70 for one class, 0.25 for another).

This "calibrated" teacher from TweetEval, despite having lower overall accuracy, produced **better-performing student models** through distillation. The softer probability distributions provided a richer training signal for the students to learn from, highlighting the importance of teacher calibration in knowledge distillation.

## Project Structure

The project is organized as follows:

```
├── README.md
├── requirements.txt
├── results/                # Stores all outputs from experiments
│   ├── benchmarks/         # CSV files from inference speed/memory benchmarks.
│   ├── distilbert_stats/   # Logs, checkpoints, and metrics for DistilBERT.
│   └── ...                 # (similar folders for other student models)
└── src/
    ├── training/           # Scripts for training and distillation.
    │   ├── fine_tuning_teacher_imdb.py
    │   └── distillation.py
    ├── inference/          # Scripts for benchmarking and quantization.
    │   ├── inference.py
    │   └── quantize_model.py
    └── utils/              # Utility functions for data, metrics, etc.
```

## How to Use

### 1. Installation

First, clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

### 2. Running the Pipeline

The project follows a standard distillation pipeline:

**Step 1: Fine-Tune the Teacher Model**
(This step is usually done once per dataset.)

```bash
# For IMDb
python src/training/fine_tuning_teacher_imdb.py

# For TweetEval
python src/training/fine_tuning_teacher_tweeteval.py
```

**Step 2: Distill Knowledge into a Student Model**
Run distillation for a specific student model or for all of them.

```bash
# Distill TinyBERT on the IMDb dataset
python src/training/distillation.py --model tinybert --dataset imdb --teacher_path results/fine_tuned_roberta_large_imdb

# Distill all students on the TweetEval dataset
python src/training/distillation.py --model all --dataset tweeteval --teacher_path results/fine_tuned_roberta_large_tweeteval_sentiment
```

**Step 3: Benchmark the Distilled Models**
Measure the inference speed and memory usage of the resulting models.

```bash
# Benchmark IMDb models
python src/inference/inference.py

# Benchmark TweetEval models
python src/inference/inference_tweeteval.py
```

**Step 4 (Optional): Post-Training Quantization (PTQ)**
Further optimize a distilled model by converting it to ONNX and quantizing it to INT8.

```bash
# Quantize the distilled TinyBERT model for IMDb
python src/inference/quantize_model.py --model_path results/distilled_tinybert_imdb_best
```

**Step 5 (Optional): Benchmark Quantized Models**
Measure the performance of the ONNX INT8 models.

```bash
python src/inference/inference_onnx.py
```