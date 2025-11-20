import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

def analyze_confidence(teacher_path, dataset_name, task_name, max_length=512, batch_size=32):
    """
    Load a model and analyze its confidence on a test set.
    The confidence of a model is defined by the mean of its maximum softmax probabilities over all test samples.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*80)
    print("Analyzing Model Confidence")
    print("="*80)
    print(f"  Model       : {teacher_path}")
    print(f"  Dataset      : {dataset_name} (Task: {task_name})")
    print(f"  Device       : {device}")
    
    print("\Loading model...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(teacher_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        model.eval()
        print("Succesfully loaded model.")
    except Exception as e:
        print(f"ERROR: Can't load model from '{teacher_path}'.")
        print(f"Detail: {e}")
        return

    print(f"Loading dataset {dataset_name.upper()}...")
    try:
        if dataset_name == 'imdb':
            dataset = load_dataset("imdb")
            test_ds = dataset['test']
        elif dataset_name == 'tweeteval':
            dataset = load_dataset("tweet_eval", task_name)
            test_ds = dataset['test']
        else:
            raise ValueError(f"Dataset '{dataset_name}' unsuported.")
        
        print(f"Dataset loaded: {len(test_ds)} test samples.")
    except Exception as e:
        print(f"ERROR: Can't load dataset from '{dataset_name}'.")
        print(f"Detail: {e}")
        return

    def collate_and_tokenize(batch):
        texts = [item['text'] for item in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return inputs

    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size,
        collate_fn=collate_and_tokenize,
        shuffle=False
    )

    print("\nBeginning confidence evaluation...")
    all_confidences = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits
            
            probabilities = F.softmax(logits, dim=-1) # No logits
            batch_confidences = torch.max(probabilities, dim=-1).values
            
            all_confidences.append(batch_confidences.cpu())

    all_confidences_tensor = torch.cat(all_confidences)
    mean_confidence = torch.mean(all_confidences_tensor).item()

    print("\n" + "="*80)
    print("CONFIDENCE RESULTS")
    print("="*80)
    print(f"  Dataset                : {dataset_name.upper()} (Task: {task_name})")
    print(f"  Teacher model              : {teacher_path}")
    print(f"  Total number of samples    : {len(all_confidences_tensor)}")
    print(f"  TEACHER AVERAGE CONFIDENCE : {mean_confidence * 100:.2f}%")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Analyzing the confidence of a model on a certain dataset")
    
    parser.add_argument('--teacher_path', type=str, required=True,
                        help='Path of the fine-tuned model to analyze.')
    parser.add_argument('--dataset', type=str, required=True, choices=['imdb', 'tweeteval'],
                        help='Dataset to use for evaluation (imdb ou tweeteval).')
    parser.add_argument('--task', type=str, default='sentiment', choices=['sentiment', 'hate'],
                        help='TweetEval specific task (default: sentiment).')
    
    args = parser.parse_args()

    MAX_LENGTH = 512
    BATCH_SIZE = 32

    analyze_confidence(
        args.teacher_path, 
        args.dataset, 
        args.task, 
        MAX_LENGTH,
        BATCH_SIZE
    )

if __name__ == "__main__":
    main()
