# Knowledge Distillation for Sentiment Analysis

This project demonstrates the process of using knowledge distillation to create a smaller, faster sentiment analysis model from a large, fine-tuned "teacher" model. The primary goal is to transfer the performance of a large model to a more lightweight one without a significant loss in accuracy.

## Project Overview

The project is divided into three main parts:

1.  **Fine-tuning the Teacher Model**: A large, pre-trained language model, `roberta-large`, is fine-tuned on the IMDb movie review dataset for sentiment analysis. This creates a powerful and accurate "teacher" model.
2.  **Knowledge Distillation (Future Work)**: The knowledge from the fine-tuned teacher model will be "distilled" into a smaller "student" model. This process trains the student to mimic the teacher's output, effectively transferring its capabilities.
3.  **Inference**: The final model can be used to predict the sentiment of new text.

## The "Teacher" Model

The teacher model is a `roberta-large` model that has been fine-tuned on the IMDb dataset for sentiment classification (positive/negative).

You can download the already fine-tuned model from the Hugging Face Hub:
[Idrisdesu/fine_tuned_roberta_large_imdb](https://huggingface.co/Idrisdesu/fine_tuned_roberta_large_imdb)

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