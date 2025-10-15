# Libraries
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Load the fine-tuned model and tokenizer
model__path = "fine_tuned_roberta_large"
model = AutoModelForSequenceClassification.from_pretrained(model__path)

# GPU
tokenizer = AutoTokenizer.from_pretrained(model__path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict_sentiment(sentence):
  model.eval()
  inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
  inputs = {k: v.to(device) for k, v in inputs.items()}
  with torch.no_grad():
      outputs = model(**inputs)

  logits = outputs.logits
  probabilities = torch.nn.functional.softmax(logits,dim=-1)
  predicted_class = torch.argmax(probabilities,dim=-1).item()

  label_map = {0:"negative",1:"positive"}
  return label_map[predicted_class]


if __name__ == "__main__":
    # Test on a few sentences
    sentence1 = "This is incredible, I love it !"
    prediction1 = predict_sentiment(sentence1)
    print(f"Sentence: {sentence1}\nPredicted Sentiment: {prediction1}\n")
    sentence2 = "This movie was so incredibly bad"
    prediction2 = predict_sentiment(sentence2)
    print(f"Sentence: {sentence2}\nPredicted Sentiment: {prediction2}\n")
    sentence3 = "Don't pay attention to people if they say it's not good"
    prediction3 = predict_sentiment(sentence3)
    print(f"Sentence: {sentence3}\nPredicted Sentiment: {prediction3}\n")