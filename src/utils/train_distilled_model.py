import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
from src.utils.data_utils import (
    load_and_prepare_dataset_imdb, 
    load_and_prepare_dataset_tweeteval
)
import argparse

STUDENT_MODELS = {
    'tinybert': 'huawei-noah/TinyBERT_General_4L_312D',
    'minilm': 'microsoft/MiniLM-L12-H384-uncased',
    'distilbert': 'distilbert-base-uncased',
    'distilroberta': 'distilroberta-base',
}

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, teacher_tokenizer, device='cuda', use_amp=True):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.teacher_tokenizer = teacher_tokenizer
        self.device = device
        self.use_amp = use_amp
        
        self.scaler = GradScaler() if use_amp else None
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
        """
        Computes the distillation loss (soft KL loss + hard CrossEntropy loss)
        """
        # Soft Loss (KL Divergence)
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        # Hard Loss (Cross Entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Distillation loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        return loss, soft_loss.item(), hard_loss.item()
    
    def train_epoch(self, train_loader, optimizer, scheduler, temperature=2.0, alpha=0.5, debug_print=False):
        """
        Train the student for one epoch
        """
        self.student.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="  Training Epoch", leave=False)
        for i, batch in enumerate(progress_bar):
            student_input_ids = batch['input_ids'].to(self.device)
            student_attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels', batch.get('label')).to(self.device)
            raw_texts = batch['text']

            # Tokenization
            teacher_inputs = self.teacher_tokenizer(
                raw_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with autocast(enabled=self.use_amp):
                with torch.no_grad():
                    teacher_outputs = self.teacher(**teacher_inputs)
                    teacher_logits = teacher_outputs.logits
                
                student_outputs = self.student(
                    input_ids=student_input_ids, 
                    attention_mask=student_attention_mask
                )
                student_logits = student_outputs.logits

                if debug_print and i == 0:
                    try:
                        # Teacher's confidence scores
                        teacher_raw_probs = F.softmax(teacher_logits, dim=-1)
                        tqdm.write("\n--- DEBUG: COMPARING LABELS ---")
                        for j in range(min(5, len(labels))): # Print the first 5 samples
                            tqdm.write(f"  Sample {j}:")
                            tqdm.write(f"    Text: {raw_texts[j][:100]}...")
                            tqdm.write(f"    Hard Label (truth): {labels[j].item()}")
                            tqdm.write(f"    Teacher Logits (confidence): {teacher_raw_probs[j].detach().cpu().numpy()}")
                        tqdm.write("--- END DEBUG ---\n")
                    except Exception as e:
                        tqdm.write(f"\n--- DEBUG: Display Error: {e} ---\n")

               # Compute distillation loss
                loss, soft_loss, hard_loss = self.distillation_loss(
                    student_logits, teacher_logits, labels, 
                    temperature=temperature, alpha=alpha
                )
            
            optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader, desc="Evaluating"):
        """
        Evaluate the student model
        """
        self.student.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        progress_bar = tqdm(eval_loader, desc=f"  {desc}", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch.get('labels', batch.get('label')).to(self.device)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    loss = F.cross_entropy(outputs.logits, labels)
                
                total_loss += loss.item()
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(eval_loader)
        return accuracy, avg_loss

def main():
    parser = argparse.ArgumentParser(description='Final Distillation Training Script')
    
    parser.add_argument('--model', type=str, required=True, choices=STUDENT_MODELS.keys(),
                        help='Student model.')
    parser.add_argument('--dataset', type=str, required=True, choices=['imdb', 'tweeteval'],
                        help='Dataset.')
    parser.add_argument('--task', type=str, default='sentiment', choices=['sentiment', 'hate'],
                        help='TweetEval task.')
    parser.add_argument('--teacher_path', type=str, required=True,
                        help='Path to the fine-tuned teacher model.')
    
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP.')
    
    args = parser.parse_args()

    # Hyperparameters
    config = {
        # Distillation hyperparameters
        'learning_rate': 2e-5,
        'temperature': 1.5,
        'alpha': 0.7,
        
        # Training parameters
        'epochs': 5,
        'batch_size': 16,
        'val_batch_size': 16,
        
        # Data parameters
        'max_length': 128  # 128 for tweeteval, 384 for imdb
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and torch.cuda.is_available()
    save_path = f"results/distilled_{args.model}_{args.dataset}_best"

    print("\n" + "="*80)
    print("BEGIN DISTILLATION TRAINING")
    print("="*80)
    print(f"  Student     : {args.model}")
    print(f"  Dataset      : {args.dataset} (Task: {args.task})")
    print(f"  Professor   : {args.teacher_path}")
    print(f"  Device       : {device} | AMP: {use_amp}")
    print(f"  Saving at   : {save_path}")
    print(f"  Hyperparams  : LR={config['learning_rate']}, Temp={config['temperature']}, Alpha={config['alpha']}")
    print(f"  Training : {config['epochs']} époques, Batch Size={config['batch_size']}")
    print_cuda_info()

    print(f"\n{'='*80}\nLoading models...")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    num_labels = teacher_model.config.num_labels
    print(f"Professor loaded. Number of labels: {num_labels}")

    student_model_name = STUDENT_MODELS[args.model]
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_name,
        num_labels=num_labels,
        trust_remote_code=True # Doesn't compile without this for some models
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    print(f"Student '{args.model}' loaded.")

    print(f"\n{'='*80}\nLoading Dataset {args.dataset.upper()}...")
    if args.dataset == 'imdb':
        _, train_ds, val_ds, test_ds, data_collator = load_and_prepare_dataset_imdb(
            model_name=student_model_name,
            max_length=config['max_length'],
            batch_size=config['batch_size']
        )
    elif args.dataset == 'tweeteval':
        _, train_ds, val_ds, test_ds, data_collator = load_and_prepare_dataset_tweeteval(
            task_name=args.task,
            model_name=student_model_name,
            max_length=config['max_length'],
            batch_size=config['batch_size']
        )
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_ds, batch_size=config['val_batch_size'], collate_fn=data_collator)
    test_loader = DataLoader(test_ds, batch_size=config['val_batch_size'], collate_fn=data_collator)
    print(f"Dataset loaded. {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test.")

    trainer = DistillationTrainer(
        teacher_model.to(device), 
        student_model.to(device), 
        teacher_tokenizer, 
        device, 
        use_amp
    )

    optimizer = AdamW(student_model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10, # 10% warmup
        num_training_steps=total_steps
    )

    print(f"\n{'='*80}\nBeginning of training...")
    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        
        train_loss = trainer.train_epoch(
            train_loader, optimizer, scheduler,
            config['temperature'], config['alpha'],
            debug_print=True  # Print labels
        )
        
        # Validation
        val_accuracy, val_loss = trainer.evaluate(val_loader, desc="Validating")
        
        print(f"  Époque {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}%")
        
        # Save the best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            print(f"New best validation accuracy. Saving model at: {save_path}")
            student_model.save_pretrained(save_path)
            student_tokenizer.save_pretrained(save_path)

    print(f"\n{'='*80}\nTraining over.")
    print(f"Loading the best model from {save_path} for the final evaluation on the testing set...")
    
    final_student_model = AutoModelForSequenceClassification.from_pretrained(
        save_path,
        trust_remote_code=True
    ).to(device)

    final_trainer = DistillationTrainer(
        teacher_model.to(device), 
        final_student_model, 
        teacher_tokenizer, 
        device, 
        use_amp
    )

    test_accuracy, test_loss = final_trainer.evaluate(test_loader, desc="Final Testing")

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"  Model       : {args.model} (distilled on {args.dataset})")
    print(f"  Best Val Acc : {best_val_acc*100:.2f}%")
    print(f"  Test Accuracy     : {test_accuracy*100:.2f}%")
    print(f"  Test Loss         : {test_loss:.4f}")
    print(f"  Saving at : {save_path}")
    print("="*80)

def print_cuda_info():
    if torch.cuda.is_available():
        print(f"  CUDA available: Yes")
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CUDA available: No (Training on CPU)")

if __name__ == "__main__":
    main()
