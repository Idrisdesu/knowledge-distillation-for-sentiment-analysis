# Libraries
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, DataCollatorWithPadding
from tqdm import tqdm
from src.utils.metrics import compute_metrics
from src.utils.data_utils import load_and_prepare_dataset_imdb

# Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Model and Tokenizer
model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2, use_auth_token=hf_token)
tokenizer, train_ds, val_ds, test_ds, data_collator = load_and_prepare_dataset_imdb(model_name="roberta-large", max_length=384, batch_size=16)

# DataLoaders
batch_size = 16 # Originally 32 but Out of Memory
train_dataloader = DataLoader(train_ds, batch_size=batch_size,shuffle=True,collate_fn=data_collator)
eval_dataloader = DataLoader(val_ds, batch_size=batch_size,collate_fn=data_collator)

# Optimizer
optimizer = AdamW(model.parameters(),lr=2e-5)

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
for epoch in range(1,num_epoch+1):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}
        outputs = model(**batch)
        if torch.cuda.device_count() > 1 and hasattr(model, 'module') and isinstance(model, nn.DataParallel):
            loss = outputs.loss.mean()  # For DataParallel
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
model_dir = 'results/fine_tuned_roberta_large'
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
