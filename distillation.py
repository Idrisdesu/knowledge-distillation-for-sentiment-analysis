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
import numpy as np
import csv
import os
from datetime import datetime
from data_utils import load_and_prepare_dataset_imdb
import argparse
import json
from itertools import product
import multiprocessing as mp
from queue import Empty
import time

# Student model options
STUDENT_MODELS = {
    'tinybert': 'huawei-noah/TinyBERT_General_4L_312D',
    'minilm': 'microsoft/MiniLM-L12-H384-uncased',
    'distilbert': 'distilbert-base-uncased',
    'distilroberta': 'distilroberta-base',
}

# Hyperparameter search spaces
HYPERPARAM_GRID = {
    'temperature': [1.5, 2.0, 3.0, 4.0],
    'alpha': [0.5, 0.7, 0.9],
    'learning_rate': [1e-5, 2e-5, 5e-5]
}


class DistillationTrainer:
    def __init__(self, teacher_model, student_model, teacher_tokenizer, device='cuda', use_amp=True):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.teacher_tokenizer = teacher_tokenizer
        self.device = device
        self.use_amp = use_amp
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        return loss, soft_loss.item(), hard_loss.item()
    
    def train_epoch(self, train_loader, optimizer, scheduler, temperature=2.0, alpha=0.5):
        self.student.train()
        total_loss = 0
        total_soft_loss = 0
        total_hard_loss = 0
        
        for batch in train_loader:
            student_input_ids = batch['input_ids'].to(self.device)
            student_attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels', batch.get('label')).to(self.device)

            raw_texts = batch['text']
            teacher_inputs = self.teacher_tokenizer(
                raw_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Use mixed precision
            with autocast(enabled=self.use_amp):
                with torch.no_grad():
                    teacher_outputs = self.teacher(**teacher_inputs)
                    teacher_logits = teacher_outputs.logits
                
                student_outputs = self.student(
                    input_ids=student_input_ids, 
                    attention_mask=student_attention_mask
                )
                student_logits = student_outputs.logits
                
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
            total_soft_loss += soft_loss
            total_hard_loss += hard_loss
        
        n = len(train_loader)
        return total_loss / n, total_soft_loss / n, total_hard_loss / n
    
    def evaluate(self, eval_loader):
        self.student.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_loader:
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


def train_with_hyperparams_worker(gpu_id, task_queue, result_queue, teacher_path, 
                                   shared_args, use_amp=True):
    """Worker process for parallel training on specific GPU"""
    
    # Set device for this worker
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    
    # Load teacher model once per worker
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher_model.to(device)
    teacher_model.eval()
    
    while True:
        try:
            task = task_queue.get(timeout=1)
        except Empty:
            continue
            
        if task is None:  # Poison pill
            break
        
        student_choice, temp, alpha, lr, num_epochs, patience, subset_ratio = task
        
        try:
            # Load student model
            student_model_name = STUDENT_MODELS[student_choice]
            student_model = AutoModelForSequenceClassification.from_pretrained(
                student_model_name,
                num_labels=teacher_model.config.num_labels
            )
            student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
            
            # Load dataset with subset
            _, train_ds, val_ds, test_ds, data_collator = load_and_prepare_dataset_imdb(
                model_name=student_model_name,
                max_length=256,
                batch_size=16
            )
            
            # Use subset for faster search
            if subset_ratio < 1.0:
                train_size = int(len(train_ds) * subset_ratio)
                indices = torch.randperm(len(train_ds))[:train_size].tolist()
                train_ds = train_ds.select(indices)
            
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=data_collator)
            val_loader = DataLoader(val_ds, batch_size=32, collate_fn=data_collator)
            
            # Initialize trainer
            trainer = DistillationTrainer(teacher_model, student_model, teacher_tokenizer, device, use_amp)
            
            # Optimizer and scheduler
            optimizer = AdamW(student_model.parameters(), lr=lr, weight_decay=0.01)
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=total_steps // 10,
                num_training_steps=total_steps
            )
            
            # Training loop with early stopping
            best_val_acc = 0.0
            epochs_without_improvement = 0
            
            for epoch in range(num_epochs):
                train_loss, soft_loss, hard_loss = trainer.train_epoch(
                    train_loader, optimizer, scheduler, temp, alpha
                )
                val_accuracy, val_loss = trainer.evaluate(val_loader)
                
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if patience is not None and epochs_without_improvement >= patience:
                    break
            
            result = {
                'student_choice': student_choice,
                'temperature': temp,
                'alpha': alpha,
                'learning_rate': lr,
                'val_accuracy': best_val_acc,
                'epochs_trained': epoch + 1,
                'success': True,
                'gpu_id': gpu_id
            }
            
            # Clean up
            del student_model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()
            
        except Exception as e:
            result = {
                'student_choice': student_choice,
                'temperature': temp,
                'alpha': alpha,
                'learning_rate': lr,
                'error': str(e),
                'success': False,
                'gpu_id': gpu_id
            }
        
        result_queue.put(result)


def successive_halving_search(student_choice, teacher_path, hyperparam_grid, 
                              num_gpus=2, use_amp=True):
    """
    Successive Halving Grid Search with Multi-GPU parallelization
    
    Round 1: Train all configs for 1 epoch with 30% data -> Keep top 50%
    Round 2: Train top 50% for 2 epochs with 50% data -> Keep top 50%
    Round 3: Train top 25% for 3 epochs with 100% data -> Return best
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 SUCCESSIVE HALVING SEARCH: {student_choice.upper()}")
    print(f"   Strategy: Multi-GPU Parallel + Early Stopping + AMP")
    print(f"   GPUs: {num_gpus}")
    print(f"{'='*80}")
    
    # Generate all combinations
    param_combinations = list(product(
        hyperparam_grid['temperature'],
        hyperparam_grid['alpha'],
        hyperparam_grid['learning_rate']
    ))
    
    total_combinations = len(param_combinations)
    print(f"📊 Total combinations: {total_combinations}")
    
    all_results = []
    
    # Round 1: Quick screening with 1 epoch, 30% data
    print(f"\n🔍 Round 1: Screening {total_combinations} configs")
    print(f"   Settings: 1 epoch, 30% data, patience=1")
    
    round1_results = parallel_train_configs(
        student_choice=student_choice,
        teacher_path=teacher_path,
        param_combinations=param_combinations,
        num_epochs=1,
        patience=1,
        subset_ratio=0.3,
        num_gpus=num_gpus,
        use_amp=use_amp,
        round_name="Round 1"
    )
    
    all_results.extend(round1_results)
    
    # Keep top 50%
    top_half = sorted(round1_results, key=lambda x: x['val_accuracy'], reverse=True)
    top_half = top_half[:max(1, len(top_half) // 2)]
    top_half_combos = [(r['temperature'], r['alpha'], r['learning_rate']) for r in top_half]
    
    print(f"   ✓ Top 50%: {len(top_half)} configs kept")
    print(f"   Best so far: {top_half[0]['val_accuracy']*100:.2f}%")
    
    # Round 2: Medium training with 2 epochs, 50% data
    print(f"\n🔍 Round 2: Refining {len(top_half)} configs")
    print(f"   Settings: 2 epochs, 50% data, patience=1")
    
    round2_results = parallel_train_configs(
        student_choice=student_choice,
        teacher_path=teacher_path,
        param_combinations=top_half_combos,
        num_epochs=2,
        patience=1,
        subset_ratio=0.5,
        num_gpus=num_gpus,
        use_amp=use_amp,
        round_name="Round 2"
    )
    
    all_results.extend(round2_results)
    
    # Keep top 50% again
    top_quarter = sorted(round2_results, key=lambda x: x['val_accuracy'], reverse=True)
    top_quarter = top_quarter[:max(1, len(top_quarter) // 2)]
    top_quarter_combos = [(r['temperature'], r['alpha'], r['learning_rate']) for r in top_quarter]
    
    print(f"   ✓ Top 50%: {len(top_quarter)} configs kept")
    print(f"   Best so far: {top_quarter[0]['val_accuracy']*100:.2f}%")
    
    # Round 3: Final training with 3 epochs, 100% data
    print(f"\n🔍 Round 3: Final evaluation of {len(top_quarter)} configs")
    print(f"   Settings: 3 epochs, 100% data, patience=2")
    
    round3_results = parallel_train_configs(
        student_choice=student_choice,
        teacher_path=teacher_path,
        param_combinations=top_quarter_combos,
        num_epochs=3,
        patience=2,
        subset_ratio=1.0,
        num_gpus=num_gpus,
        use_amp=use_amp,
        round_name="Round 3"
    )
    
    all_results.extend(round3_results)
    
    # Find best overall
    best_result = max(round3_results, key=lambda x: x['val_accuracy'])
    best_params = {
        'temperature': best_result['temperature'],
        'alpha': best_result['alpha'],
        'learning_rate': best_result['learning_rate']
    }
    best_accuracy = best_result['val_accuracy']
    
    print(f"\n{'='*80}")
    print(f"✅ SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"🏆 Best Hyperparameters:")
    print(f"   Temperature: {best_params['temperature']}")
    print(f"   Alpha: {best_params['alpha']}")
    print(f"   Learning Rate: {best_params['learning_rate']}")
    print(f"   Best Val Accuracy: {best_accuracy*100:.2f}%")
    print(f"\n📈 Search efficiency:")
    print(f"   Evaluated {total_combinations} initial configs")
    print(f"   Total training runs: {len(all_results)}")
    print(f"   Time saved vs full grid: ~{(total_combinations * 3) / len(all_results):.1f}x")
    
    return best_params, best_accuracy, all_results


def parallel_train_configs(student_choice, teacher_path, param_combinations, 
                           num_epochs, patience, subset_ratio, num_gpus, use_amp, round_name):
    """Execute parallel training across multiple GPUs"""
    
    # Create multiprocessing queues
    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    # Fill task queue
    for temp, alpha, lr in param_combinations:
        task_queue.put((student_choice, temp, alpha, lr, num_epochs, patience, subset_ratio))
    
    # Add poison pills
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # Start worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=train_with_hyperparams_worker,
            args=(gpu_id, task_queue, result_queue, teacher_path, {}, use_amp)
        )
        p.start()
        processes.append(p)
    
    # Collect results with progress bar
    results = []
    progress_bar = tqdm(
        total=len(param_combinations),
        desc=f"  {round_name}",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    best_in_round = 0.0
    completed = 0
    
    while completed < len(param_combinations):
        try:
            result = result_queue.get(timeout=1)
        except Empty:
            continue
        
        if result['success']:
            results.append(result)
            
            if result['val_accuracy'] > best_in_round:
                best_in_round = result['val_accuracy']
                progress_bar.set_postfix({
                    'Best': f"{best_in_round*100:.1f}%",
                    'GPU': result['gpu_id']
                })
        else:
            tqdm.write(f"⚠️  Failed on GPU {result['gpu_id']}: {result['error'][:50]}")
        
        completed += 1
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    return results


def final_training_with_best_params(student_choice, teacher_path, device, 
                                    best_params, num_epochs=5, use_amp=True):
    """Final training with best hyperparameters on full data"""
    
    print(f"\n{'='*80}")
    print(f"🎓 FINAL TRAINING: {student_choice.upper()}")
    print(f"{'='*80}")
    print(f"⚙️  Best hyperparameters:")
    print(f"   Temperature: {best_params['temperature']}")
    print(f"   Alpha: {best_params['alpha']}")
    print(f"   Learning Rate: {best_params['learning_rate']}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Mixed Precision: {use_amp}")
    
    # Load models
    student_model_name = STUDENT_MODELS[student_choice]
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_name,
        num_labels=2  # Binary classification for IMDB
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    
    # Load full dataset
    _, train_ds, val_ds, test_ds, data_collator = load_and_prepare_dataset(
        model_name=student_model_name,
        max_length=256,
        batch_size=16
    )
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=data_collator)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=data_collator)
    
    # Initialize trainer
    trainer = DistillationTrainer(teacher_model, student_model, teacher_tokenizer, device, use_amp)
    
    # Optimizer and scheduler
    optimizer = AdamW(student_model.parameters(), lr=best_params['learning_rate'], weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_acc = 0.0
    epoch_results = []
    
    progress_bar = tqdm(range(num_epochs), desc=f"  📚 Training")
    for epoch in progress_bar:
        train_loss, soft_loss, hard_loss = trainer.train_epoch(
            train_loader, optimizer, scheduler, 
            best_params['temperature'], best_params['alpha']
        )
        val_accuracy, val_loss = trainer.evaluate(val_loader)
        
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.3f}',
            'val_acc': f'{val_accuracy*100:.1f}%',
            'best': f'{best_val_acc*100:.1f}%'
        })
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_path = f"./distilled_{student_choice}_imdb_best"
            student_model.save_pretrained(save_path)
            student_tokenizer.save_pretrained(save_path)
            tqdm.write(f"   💾 Saved checkpoint: {val_accuracy*100:.2f}%")
        
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy
        })
    
    # Test evaluation
    print(f"\n🧪 Final Test Evaluation:")
    test_accuracy, test_loss = trainer.evaluate(test_loader)
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    return {
        'model_name': student_choice,
        'best_params': best_params,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'epoch_results': epoch_results
    }


def save_results(student_choice, all_results, best_params, best_accuracy):
    """Save search results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save all results
    results_file = f'hypersearch_{student_choice}_{timestamp}.csv'
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Temperature', 'Alpha', 'Learning Rate', 'Val Accuracy (%)', 'Epochs Trained'])
        
        for result in sorted(all_results, key=lambda x: x['val_accuracy'], reverse=True):
            writer.writerow([
                result['temperature'],
                result['alpha'],
                result['learning_rate'],
                f"{result['val_accuracy']*100:.2f}",
                result.get('epochs_trained', 'N/A')
            ])
    
    print(f"   📄 Results saved: {results_file}")
    
    # Save best params
    params_file = f'best_params_{student_choice}_{timestamp}.json'
    with open(params_file, 'w') as f:
        json.dump({
            'model': student_choice,
            'best_params': best_params,
            'best_val_accuracy': float(best_accuracy),
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"   📄 Best params saved: {params_file}")


def main():
    parser = argparse.ArgumentParser(description='Fast Knowledge Distillation with Optimized Search')
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['tinybert', 'minilm', 'distilbert', 'distilroberta', 'all'],
        required=True,
        help='Student model to train (or "all")'
    )
    parser.add_argument('--final_epochs', type=int, default=5, 
                       help='Epochs for final training')
    parser.add_argument('--teacher_path', type=str, default='./fine_tuned_roberta_large_imdb', 
                       help='Path to teacher model')
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    
    args = parser.parse_args()
    
    # Setup
    num_gpus = args.gpus if args.gpus else torch.cuda.device_count()
    use_amp = not args.no_amp
    device = torch.device('cuda:0')
    
    print("\n" + "="*80)
    print("🚀 OPTIMIZED KNOWLEDGE DISTILLATION")
    print("="*80)
    print(f"💻 GPUs available: {torch.cuda.device_count()}")
    print(f"💻 GPUs to use: {num_gpus}")
    print(f"⚡ Mixed Precision: {use_amp}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine models to train
    if args.model == 'all':
        models_to_train = list(STUDENT_MODELS.keys())
    else:
        models_to_train = [args.model]
    
    print(f"🎯 Models: {', '.join(m.upper() for m in models_to_train)}")
    
    all_results = {}
    overall_start = time.time()
    
    for student_choice in models_to_train:
        print(f"\n\n{'#'*80}")
        print(f"   PROCESSING: {student_choice.upper()}")
        print(f"{'#'*80}")
        
        model_start = time.time()
        
        # Hyperparameter search with successive halving
        best_params, best_accuracy, search_results = successive_halving_search(
            student_choice=student_choice,
            teacher_path=args.teacher_path,
            hyperparam_grid=HYPERPARAM_GRID,
            num_gpus=num_gpus,
            use_amp=use_amp
        )
        
        # Save search results
        save_results(student_choice, search_results, best_params, best_accuracy)
        
        # Final training
        final_result = final_training_with_best_params(
            student_choice=student_choice,
            teacher_path=args.teacher_path,
            device=device,
            best_params=best_params,
            num_epochs=args.final_epochs,
            use_amp=use_amp
        )
        
        all_results[student_choice] = final_result
        
        model_time = time.time() - model_start
        print(f"\n✅ {student_choice.upper()} completed in {model_time/60:.1f} minutes")
        print(f"   Test Accuracy: {final_result['test_accuracy']*100:.2f}%")
    
    # Final summary
    total_time = time.time() - overall_start
    print(f"\n\n{'='*80}")
    print("🎉 ALL TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"\n📊 Results Summary:")
    print(f"{'─'*80}")
    
    for model_name, result in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Hyperparams: T={result['best_params']['temperature']}, "
              f"α={result['best_params']['alpha']}, LR={result['best_params']['learning_rate']}")
        print(f"  Val Acc: {result['best_val_accuracy']*100:.2f}%")
        print(f"  Test Acc: {result['test_accuracy']*100:.2f}%")
    
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Required for multiprocessing on CUDA
    mp.set_start_method('spawn', force=True)

    main()
