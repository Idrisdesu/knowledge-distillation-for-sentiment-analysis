# Libraries
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from tqdm import tqdm
from metrics import compute_metrics
from data_utils import load_and_prepare_dataset_tweeteval 

# Hugging Face token
hf_token = os.getenv("HF_TOKEN")

TASK_NAME = "sentiment" # The task
TEACHER_MODEL_NAME = "roberta-large"
MODEL_SAVE_DIR = f'fine_tuned_roberta_large_tweeteval_{TASK_NAME}'

print(f"Loading TweetEval dataset for the task: {TASK_NAME}")
tokenizer, train_ds, val_ds, test_ds, data_collator = load_and_prepare_dataset_tweeteval(
    task_name=TASK_NAME,
    model_name=TEACHER_MODEL_NAME, 
    max_length=128, # The tweets are way shorter than IMDb reviews
    batch_size=16
)
model = AutoModelForSequenceClassification.from_pretrained(
    TEACHER_MODEL_NAME, 
    num_labels=3, # 3 if TASK_NAME == "sentiment" else 2
    use_auth_token=hf_token
)

# DataLoaders
batch_size = 32 # We can increase batch size since tweets are short
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(val_ds, batch_size=batch_size, collate_fn=data_collator)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epoch = 3
num_training_steps = num_epoch * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.to(device)

# Training
model.train()
for epoch in range(1, num_epoch + 1):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}
        outputs = model(**batch)
        if torch.cuda.device_count() > 1 and hasattr(model, 'module') and isinstance(model, nn.DataParallel):
            loss = outputs.loss.mean()
        else:
            loss = outputs.loss

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({"loss":loss.item()})

    # Benchmarking
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.append(preds)
            all_labels.append(batch['labels'])

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_preds, all_labels)
    print(f"Epoch {epoch} validation metrics: {metrics}")

    model.train()

# Save the model
print(f"Model saved in : {MODEL_SAVE_DIR}")
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
