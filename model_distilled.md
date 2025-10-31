# 🧠 Distilled Models – IMDB Sentiment Classification

This document summarizes the distilled models trained for **binary sentiment analysis** on the **IMDB dataset**.  
Each model was distilled from a larger transformer to achieve a better trade-off between **accuracy** and **model size**.

---

## 📊 Model Overview

| Model | Parameters | Test Accuracy | Hugging Face Repository |
|--------|-------------|----------------|--------------------------|
| **DistilRoBERTa (IMDB)** | ~82M | **92.80%** | [🔗 Idrisdesu/distilled_distilroberta_imdb](https://huggingface.co/Idrisdesu/distilled_distilroberta_imdb) |
| **DistilBERT (IMDB)** | ~66M | **91.64%** | [🔗 youssefennouri/distilled_distilbert_imdb](https://huggingface.co/youssefennouri/distilled_distilbert_imdb) |
| **MiniLM (IMDB)** | ~33M | **91.98%** | [🔗 youssefennouri/distilled_minilm_imdb](youssefennouri/distilled_minilm_imdb) |
| **TinyBERT (IMDB)** | ~14M | **88.24%** | [🔗 youssefennouri/distilled_tinybert_imdb](https://huggingface.co/youssefennouri/distilled_tinybert_imdb) |

---

## ⚙️ Task Details
- **Dataset:** IMDB Movie Reviews  
- **Task:** Sentiment Classification (Positive / Negative)  
- **Metric:** Accuracy on IMDB test set  
- **Distillation Objective:** Transfer knowledge from a large teacher model into smaller, faster student models.  

---

## 🚀 Usage Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "youssef-en/distilroberta-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "The movie was surprisingly good and emotional."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()
print("Positive" if prediction == 1 else "Negative")
