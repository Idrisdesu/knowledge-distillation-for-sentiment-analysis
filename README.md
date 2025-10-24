# Knowledge Distillation for Sentiment Analysis

This project demonstrates the process of using knowledge distillation to create a smaller, faster sentiment analysis model from a large, fine-tuned "teacher" model. The primary goal is to transfer the performance of a large model to a more lightweight one without a significant loss in accuracy.

## Project Overview

The project is divided into three main parts:

1.  **Fine-tuning the Teacher Model**: A large, pre-trained language model, `roberta-large`, is fine-tuned on the IMDb movie review dataset for sentiment analysis. This creates a powerful and accurate "teacher" model.
2.  **Knowledge Distillation (Future Work)**: The knowledge from the fine-tuned teacher model will be "distilled" into a smaller "student" model. This process trains the student to mimic the teacher's output, effectively transferring its capabilities.
3.  **Inference**: The final model can be used to predict the sentiment of new text.

## The "Teacher" Model

The teacher model is a `roberta-large` model that has been fine-tuned on the IMDb dataset for sentiment classification (positive/negative).

## Student Models

We trained four different student models:

| **Student Model** | **Number of Parameters** |
|--------------------|--------------------------|
| distilRoBERTa      | 82M                     |
| distilBERT         | 66M                     |
| MiniLM             | 33M                     |
| TinyBERT           | 14.5M                   |

*Table 1 – List and size of the student models.*

---

## Student Models Performance Comparison

The following table compares the performance and size of the student models (trained with response-based distillation) against the teacher model, **RoBERTa Large**.

| **Model**         | **Accuracy (%)** | **Compression Ratio** |
|--------------------|------------------|------------------------|
| RoBERTa Large      | 95.88            | 1.00                   |
| distilRoBERTa      | 92.80            | 4.33                   |
| distilBERT         | 91.64            | 5.31                   |
| MiniLM             | 91.98            | 10.76                  |
| TinyBERT           | 88.24            | 24.48                  |

*Table 2 – Comparison of student models with the teacher model (RoBERTa Large).*

## Model Links

| **Model** | **Hugging Face Repository** |
|------------|------------------------------|
| RoBERTa Large | [Idrisdesu/fine_tuned_roberta_large_imdb](https://huggingface.co/Idrisdesu/fine_tuned_roberta_large_imdb) |
| distilRoBERTa | [Idrisdesu/distilled_distilroberta_imdb](https://huggingface.co/Idrisdesu/distilled_distilroberta_imdb/tree/main) |
| distilBERT | [youssefennouri/distilled_distilbert_imdb](https://huggingface.co/youssefennouri/distilled_distilbert_imdb) |
| MiniLM | [youssefennouri/distilled_minilm_imdb](https://huggingface.co/youssefennouri/distilled_minilm_imdb) |
| TinyBERT | [youssefennouri/distilled_tinybert_imdb](https://huggingface.co/youssefennouri/distilled_tinybert_imdb) |

*Table 3 – Hugging Face repositories for each model.*

## How to Use

### 1. Setup

First, clone the repository and install the required dependencies. It is recommended to use a virtual environment.

```bash
git clone <repository-url>
cd knowledge-distillation-for-sentiment-analysis
pip install -r requirements.txt
```



### 2. Download the Pre-trained Teacher Model

Create a directory named `fine_tuned_roberta_large_imdb` and download the model files from the Hugging Face link above into this directory.

The directory structure should look like this:

```
knowledge-distillation-for-sentiment-analysis/
├── fine_tuned_roberta_large_imdb/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ... (other model files)
├── data_utils.py
├── fine_tuning_teacher.py
└── ... (other project files)
```

### 3. (Optional) Fine-tune Your Own Teacher Model

If you want to fine-tune the `roberta-large` model yourself, you can run the `fine_tuning_teacher.py` script. Make sure you are authenticated with Hugging Face if you are using a private model.

```bash
# You might need to set your Hugging Face token as an environment variable
export HF_TOKEN="your_hugging_face_token"

python fine_tuning_teacher.py
```

This will train the model and save it to a directory named `fine_tuned_roberta_large`.

### 4. Run Inference

To test the sentiment analysis prediction on new sentences, use the `inference.py` script.

```bash
python inference.py
```

This will load the fine-tuned model from the `fine_tuned_roberta_large_imdb` directory and print sentiment predictions for a few example sentences.
