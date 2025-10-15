# Libraries
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

def load_and_prepare_dataset(model_name="FacebookAI/roberta-large", max_length=384, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("stanfordnlp/imdb")

    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = ds["test"]

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])
    test_ds = test_ds.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    return tokenizer, train_ds, val_ds, test_ds, data_collator
