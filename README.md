# Real-Time Sentiment Analysis via Knowledge Distillation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Optimizing Large Language Models (LLMs) for real-time inference using Knowledge Distillation and Quantization.**

> **Key Achievement:** Compressed a RoBERTa-Large model into a MiniLM student, achieving **10× faster inference** while retaining **>91% of the original accuracy**, enabling real-time deployment on standard hardware.

---
# 📖 Overview

Large Language Models like RoBERTa deliver great accuracy but are too slow and heavy for **real-time applications** such as:
- live content moderation,
- on-device inference,
- real-time chatbot filtering.

This project implements a complete **Model Compression Pipeline**:

1. **Teacher Fine-Tuning** – optimizing RoBERTa-Large on IMDb/TweetEval.  
2. **Knowledge Distillation** – transferring the teacher’s knowledge to compact models (MiniLM, DistilBERT…).  
3. **Hyperparameter Optimization** – searching for the best temperature and α with Optuna.  
4. **Quantization** – converting models to ONNX and applying INT8 dynamic quantization for speed.

### 🧪 Datasets Used
- **IMDb** – binary sentiment classification (Positive/Negative)  
- **TweetEval** – 3-way sentiment (Positive / Negative / Neutral)

---

# 🧠 Distilled Models – IMDB Sentiment Classification

Below are the distilled models trained for **binary sentiment analysis** on the **IMDb dataset**.  
Each model was distilled from a larger high-performance teacher (RoBERTa-Large).

| Model | Parameters | Test Accuracy | Hugging Face Repository |
|--------|-------------|----------------|--------------------------|
| **DistilRoBERTa (IMDB)** | ~82M | **92.80%** | 🔗 https://huggingface.co/Idrisdesu/distilled_distilroberta_imdb |
| **DistilBERT (IMDB)** | ~66M | **91.64%** | 🔗 https://huggingface.co/youssefennouri/distilled_distilbert_imdb |
| **MiniLM (IMDB)** | ~33M | **91.98%** | 🔗 https://huggingface.co/youssefennouri/distilled_minilm_imdb |
| **TinyBERT (IMDB)** | ~14M | **88.24%** | 🔗 https://huggingface.co/youssefennouri/distilled_tinybert_imdb |

### 🚀 Usage Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "youssefennouri/distilled_minilm_imdb"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "The movie was surprisingly good and emotional."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

prediction = outputs.logits.argmax().item()
print("Positive" if prediction == 1 else "Negative")
```

---

# 📊 Key Results & Insights

## 1. Performance vs. Speed Trade-off

| Model | Accuracy (IMDb) | Speedup | Size |
|-------|:--------------:|:-------:|:----:|
| **RoBERTa-Large (Teacher)** | **95.88%** | 1× | ~1.4GB |
| DistilBERT | 92.5% | ~2× | ~260MB |
| **MiniLM (Best Trade-off)** | **91.2%** | **~10×** | **~120MB** |
| TinyBERT | 88.4% | ~20× | ~60MB |

*(See `results/benchmarks/` for raw logs.)*

## 2. The “Calibration” Discovery

We discovered that **Teacher Accuracy ≠ Teaching Quality**.

- **IMDb** → Teacher gave *overconfident* outputs → students inherit the overfitting.  
- **TweetEval** → Teacher produced *nuanced probabilities* → better generalization for students like DistilBERT.

> **Takeaway:** A well-calibrated teacher produces much stronger students than a high-accuracy but overconfident teacher.

---

# 📂 Project Structure

```text
.
├── src/
│   ├── training/       # Teacher fine-tuning & Knowledge Distillation
│   │   ├── fine_tuning_teacher_imdb.py
│   │   ├── fine_tuning_teacher_tweeteval.py
│   │   ├── distillation.py
│   │   └── ...
│   ├── inference/      # ONNX conversion, quantization, benchmarking
│   │   ├── inference.py
│   │   ├── inference_onnx.py
│   │   ├── quantize_model.py
│   │   └── ...
│   └── utils/
│       ├── teacher_confidence.py
│       └── ...
├── results/
│   ├── benchmarks/
│   ├── distilbert_stats/
│   ├── minilm_stats/
│   └── ...
└── requirements.txt
```

---

🚀 Installation

git clone https://github.com/votre-username/realtime-sentiment-distillation.git
cd realtime-sentiment-distillation
pip install -r requirements.txt

📥 Download Pre-trained Models

To reproduce our benchmarks immediately without training from scratch, you need to download the distilled and quantized models.

Option A: Automated Download (Recommended)
We provide a script to fetch all necessary models from Hugging Face and place them in the correct results/ structure.

python -m src.utils.download_models

(Note: If this script doesn't exist, please refer to Option B)

Option B: Manual DownloadIf you want to run the benchmarks, ensure your results/ folder looks like this. You can download the weights from the links in the "Distilled Models" section above or train them yourself using Step 2 and Step 3 in Usage.

Required structure for Benchmarking:

results/
├── distilled_model_imdb/
│   ├── distilled_distilbert_imdb/
│   ├── distilled_minilm_imdb/
│   └── ...
└── distilled_models_imdb_int8/  <-- (Generated via src.inference.quantize_model)
    ├── distilled_distilbert_imdb_int8_ptq_onnx/
    └── ...

⚠️ Important: The ONNX quantized models (_int8_ptq_onnx) are hardware-specific. We strongly recommend generating them on your own machine:
---

# 🛠 Usage (How to Run)

⚠️ **IMPORTANT:** Always run using `python -m` **from the project root**, otherwise you will get `ModuleNotFoundError`.

---

## 1. Train the Teacher

```bash
# IMDb:
python -m src.training.fine_tuning_teacher_imdb

# Or TweetEval:
# python -m src.training.fine_tuning_teacher_tweeteval
```

## 2. Knowledge Distillation (Train Students)

```bash
python -m src.training.distillation --model distilbert --dataset imdb
```

(See `distillation.py` for model choices.)

## 3. Quantization & Benchmarking

```bash
# Standard inference benchmark
python -m src.inference.inference

# Quantize to ONNX INT8
python -m src.inference.quantize_model

# Benchmark ONNX model
python -m src.inference.inference_onnx
```

---

# 📈 Methodology Details

### 1. Teacher Fine-Tuning  
The teacher sets the performance **upper bound**.

### 2. Hyperparameter Search (Optuna)
We optimized:
- **α** – balance between hard and soft labels  
- **temperature** – softmax smoothing  

Logs:  
`results/distilbert_stats/hypersearch_*.csv`

### 3. Teacher Calibration Analysis  
Using `teacher_confidence.py`, we evaluated:
- maximum softmax probability (MSP),
- entropy of predictions,
- calibration curves.

This explained why some students learned surprisingly better on TweetEval.

---

# 👥 Authors

- **Idris NECHNECH**  
- **Youssef ENNOURI**  
- **Younes OUDINA**  

*Project completed as part of the NLP Course (2025).*
