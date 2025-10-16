import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import csv
import os
from datetime import datetime
from data_utils import load_and_prepare_dataset
import argparse

# Student model options (from smallest to largest):
STUDENT_MODELS = {
    'tinybert': 'huawei-noah/TinyBERT_General_4L_312D',   # ~14.5M params
    'minilm': 'microsoft/MiniLM-L12-H384-uncased',         # ~33M params
    'distilbert': 'distilbert-base-uncased',              # ~66M params
    'distilroberta': 'distilroberta-base',                # ~82M params
}


class DistillationTrainer:
    def __init__(self, teacher_model, student_model, teacher_tokenizer, device='cuda'):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.teacher_tokenizer = teacher_tokenizer
        self.device = device
        
        # Freeze teacher model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
        """
        Compute distillation loss combining soft targets from teacher and hard labels
        
        Args:
            student_logits: logits from student model
            teacher_logits: logits from teacher model
            labels: ground truth labels
            temperature: temperature for softening probability distributions
            alpha: weight for distillation loss (1-alpha for hard label loss)
        """
        # Soft target loss (KL divergence between teacher and student)
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        return loss, soft_loss.item(), hard_loss.item()
    
    def train_epoch(self, train_loader, optimizer, scheduler, temperature=2.0, alpha=0.5):
        self.student.train()
        total_loss = 0
        total_soft_loss = 0
        total_hard_loss = 0
        
        progress_bar = tqdm(train_loader, desc="  Training", leave=False)
        for batch in progress_bar:
            # Student inputs are pre-tokenized and collated
            student_input_ids = batch['input_ids'].to(self.device)
            student_attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels', batch.get('label')).to(self.device)

            # Teacher inputs must be tokenized on the fly from raw text
            raw_texts = batch['text']
            teacher_inputs = self.teacher_tokenizer(
                raw_texts,
                padding=True,
                truncation=True,
                max_length=512, # Teacher can handle longer sequences
                return_tensors="pt"
            ).to(self.device)

            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher(**teacher_inputs)
                teacher_logits = teacher_outputs.logits
            
            # Get student predictions
            student_outputs = self.student(
                input_ids=student_input_ids, 
                attention_mask=student_attention_mask
            )
            student_logits = student_outputs.logits
            
            # Calculate distillation loss
            loss, soft_loss, hard_loss = self.distillation_loss(
                student_logits, teacher_logits, labels, 
                temperature=temperature, alpha=alpha
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_soft_loss += soft_loss
            total_hard_loss += hard_loss
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'soft': f'{soft_loss:.4f}',
                'hard': f'{hard_loss:.4f}'
            })
        
        n = len(train_loader)
        return total_loss / n, total_soft_loss / n, total_hard_loss / n
    
    def evaluate(self, eval_loader, desc="  Evaluating"):
        self.student.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=desc, leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                # Handle both 'label' and 'labels' keys
                labels = batch.get('labels', batch.get('label')).to(self.device)
                
                outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                loss = F.cross_entropy(outputs.logits, labels)
                total_loss += loss.item()
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(eval_loader)
        return accuracy, avg_loss


def train_single_model(student_choice, teacher_model, teacher_tokenizer, device, 
                       num_epochs=3, learning_rate=2e-5,
                       temperature=2.0, alpha=0.7):
    """Train a single student model"""
    
    print(f"\n{'='*80}")
    print(f"🎓 TRAINING: {student_choice.upper()}")
    print(f"{'='*80}")
    
    # Load student model
    student_model_name = STUDENT_MODELS[student_choice]
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_name,
        num_labels=teacher_model.config.num_labels
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    # Load dataset with the STUDENT's tokenizer (critical for compatibility!)
    print(f"📦 Loading dataset with {student_choice} tokenizer...")
    _, train_ds, val_ds, test_ds, data_collator = load_and_prepare_dataset(
        model_name=student_model_name,
        max_length=256,
        batch_size=16
    )
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=data_collator)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=data_collator)
    
    teacher_params = teacher_model.num_parameters()
    student_params = student_model.num_parameters()
    compression_ratio = teacher_params / student_params
    
    print(f"📊 Model Statistics:")
    print(f"   Teacher:  {teacher_params:>12,} parameters")
    print(f"   Student:  {student_params:>12,} parameters")
    print(f"   Compression: {compression_ratio:>10.2f}x smaller")
    print(f"\n📦 Dataset Statistics:")
    print(f"   Train samples: {len(train_ds):,}")
    print(f"   Val samples: {len(val_ds):,}")
    print(f"   Test samples: {len(test_ds):,}")
    print(f"\n⚙️  Hyperparameters:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Temperature: {temperature}")
    print(f"   Alpha (distillation weight): {alpha}")
    
    # Initialize trainer
    trainer = DistillationTrainer(teacher_model, student_model, teacher_tokenizer, device)
    
    # Optimizer and scheduler
    optimizer = AdamW(student_model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_acc = 0.0
    results = []
    
    for epoch in range(num_epochs):
        print(f"\n📖 Epoch {epoch + 1}/{num_epochs}")
        print(f"   {'-'*70}")
        
        train_loss, soft_loss, hard_loss = trainer.train_epoch(
            train_loader, optimizer, scheduler, temperature, alpha
        )
        val_accuracy, val_loss = trainer.evaluate(val_loader, desc="  Validating")
        
        print(f"   📈 Train Loss: {train_loss:.4f} (Soft: {soft_loss:.4f}, Hard: {hard_loss:.4f})")
        print(f"   📊 Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy*100:.2f}%", end="")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_path = f"./distilled_{student_choice}_imdb"
            student_model.save_pretrained(save_path)
            student_tokenizer.save_pretrained(save_path)
            print(f" ✓ [BEST - Saved]")
        else:
            print()
        
        # Store results
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'soft_loss': soft_loss,
            'hard_loss': hard_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
    
    # Final test evaluation
    print(f"\n🧪 Final Test Evaluation:")
    print(f"   {'-'*70}")
    test_accuracy, test_loss = trainer.evaluate(test_loader, desc="  Testing")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Summary
    print(f"\n✅ Training Complete for {student_choice.upper()}")
    print(f"   Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"   Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Model saved to: ./distilled_{student_choice}_imdb")
    
    return {
        'model_name': student_choice,
        'student_params': student_params,
        'compression_ratio': compression_ratio,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'epoch_results': results
    }


def save_results_to_csv(result, filename_prefix='distillation'):
    """Save results for a single model to CSV"""
    
    model_name = result['model_name']
    
    # Summary CSV for this model
    summary_file = f'{filename_prefix}_{model_name}_summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Parameters', 'Compression Ratio', 
            'Best Val Accuracy (%)', 'Test Accuracy (%)', 'Test Loss'
        ])
        
        writer.writerow([
            result['model_name'],
            f"{result['student_params']:,}",
            f"{result['compression_ratio']:.2f}x",
            f"{result['best_val_accuracy']*100:.2f}",
            f"{result['test_accuracy']*100:.2f}",
            f"{result['test_loss']:.4f}"
        ])
    
    print(f"   📄 Summary saved to: {summary_file}")
    
    # Detailed CSV for this model
    detail_file = f'{filename_prefix}_{model_name}_detailed.csv'
    
    with open(detail_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch', 'Train Loss', 'Soft Loss', 'Hard Loss', 
            'Val Loss', 'Val Accuracy (%)'
        ])
        
        for epoch_result in result['epoch_results']:
            writer.writerow([
                epoch_result['epoch'],
                f"{epoch_result['train_loss']:.4f}",
                f"{epoch_result['soft_loss']:.4f}",
                f"{epoch_result['hard_loss']:.4f}",
                f"{epoch_result['val_loss']:.4f}",
                f"{epoch_result['val_accuracy']*100:.2f}"
            ])
    
    print(f"   📄 Detailed log saved to: {detail_file}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Knowledge Distillation for Sentiment Analysis')
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['tinybert', 'minilm', 'distilbert', 'distilroberta'],
        required=True,
        help='Student model to train (tinybert, minilm, distilbert, or distilroberta)'
    )
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=2.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation loss weight')
    parser.add_argument('--teacher_path', type=str, default='./fine_tuned_roberta_large_imdb', 
                       help='Path to teacher model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("🚀 KNOWLEDGE DISTILLATION FOR SENTIMENT ANALYSIS")
    print("="*80)
    print(f"💻 Device: {device}")
    print(f"🎯 Selected Model: {args.model.upper()}")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load teacher model
    print(f"\n{'='*80}")
    print("📚 Loading Teacher Model (Fine-tuned RoBERTa-Large)")
    print(f"{'='*80}")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    print(f"✓ Teacher model loaded: {teacher_model.num_parameters():,} parameters")
    
    # Train selected model
    result = train_single_model(
        student_choice=args.model,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        temperature=args.temperature,
        alpha=args.alpha
    )
    
    # Save results
    print(f"\n{'='*80}")
    print("💾 Saving Results")
    print(f"{'='*80}")
    save_results_to_csv(result)
    
    print(f"\n{'='*80}")
    print("✅ DISTILLATION COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Final Results for {args.model.upper()}:")
    print(f"   Parameters: {result['student_params']:,}")
    print(f"   Compression: {result['compression_ratio']:.2f}x smaller than teacher")
    print(f"   Best Val Accuracy: {result['best_val_accuracy']*100:.2f}%")
    print(f"   Test Accuracy: {result['test_accuracy']*100:.2f}%")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎉 Model trained and saved successfully!\n")


if __name__ == "__main__":
    main()