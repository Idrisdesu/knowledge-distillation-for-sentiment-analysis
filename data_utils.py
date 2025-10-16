# Libraries
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from typing import Any, Dict, List

class CustomDataCollator(DataCollatorWithPadding):
    """
    Custom Data Collator that handles raw text columns.
    
    By default, DataCollatorWithPadding tries to pad and convert all columns
    to tensors. This fails for columns containing strings (like our 'text' column).
    This custom collator manually separates the text column, lets the parent
    class handle the tensor-based columns, and then adds the text column back.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate the raw text from other features.
        # The 'text' column is expected to be present in the features.
        texts = [feature.pop("text", None) for feature in features]
        
        # Use the parent collator to pad the numerical features (input_ids, attention_mask, labels)
        batch = super().__call__(features)
        
        # Add the raw text back to the batch.
        # This will be used for on-the-fly tokenization by the teacher model.
        batch["text"] = texts
        return batch

def load_and_prepare_dataset(model_name="FacebookAI/roberta-large", max_length=384, batch_size=16):
    """
    Load and prepare IMDB dataset for training.
    
    This version tokenizes the data for the STUDENT model but keeps the original 'text'
    column, which is needed for on-the-fly tokenization by the TEACHER model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("stanfordnlp/imdb")

    # Split train into train/val
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = ds["test"]

    def tokenize_function(examples):
        # Tokenize for the student model
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length
        )

    # Tokenize datasets. The original 'text' and 'label' columns are kept by default.
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    # Rename 'label' to 'labels' for consistency with Hugging Face models
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    
    # Remove the original text column now that tokenization is done,
    # but the mapped dataset still holds the text in memory for the collator.
    # This step is actually not strictly necessary if the collator handles it,
    # but it's good practice to clean up columns. The key is that the 'text'
    # from the original dataset is still available in the features passed to the collator.
    # Let's keep it simple and not remove columns, relying on the collator.

    # Use our custom data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer, return_tensors="pt")

    return tokenizer, train_ds, val_ds, test_ds, data_collator
