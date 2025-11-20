# Libraries
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import random
import pynvml
import csv
import os

# Initialize NVML
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming GPU is at index 0
    gpu_available = True
except pynvml.NVMLError:
    gpu_available = False
    print("NVIDIA GPU not found. Running in CPU mode. GPU metrics will not be available.")

def get_gpu_metrics(handle):
    """Returns GPU memory usage in MB and power consumption in Watts."""
    if not gpu_available:
        return 0, 0
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
    return mem_info.used / (1024**2), power_usage

def predict_sentiment_and_time(sentence, model, tokenizer, device):
    if hasattr(model, 'eval'):
        model.eval()
    
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if device.type == 'cuda':
        torch.cuda.synchronize() 
        
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    inference_time = end_time - start_time

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    label_map = {0:"negative", 1:"positive"}
    return label_map[predicted_class], inference_time

def load_sentences_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        return sentences
    except FileNotFoundError:
        print(f"Error: the file {filepath} was not found.")
        return []


if __name__ == "__main__":
    # List of model paths to benchmark if multiple
    model_paths = [
        "results/fine_tuned_roberta_large_imdb",
        "results/distilled_model_imdb/distilled_distilbert_imdb",
        "results/distilled_model_imdb/distilled_distilroberta_imdb",
        "results/distilled_model_imdb/distilled_minilm_imdb",
        "results/distilled_model_imdb/distilled_tinybert_imdb"
    ]

    monte_carlo_sentences = load_sentences_from_file("monte_carlo_sentences.txt")
    if not monte_carlo_sentences:
        print("No sentences loaded for Monte Carlo simulation. Exiting.")
        exit(1)

    N = 5000 # Number of Monte Carlo runs
    
    results = []

    for model_path in model_paths:
        if not os.path.isdir(model_path):
            print(f"Model directory not found: {model_path}. Skipping.")
            continue

        print(f"\nLoading model: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() and gpu_available else "cpu")
        model = model.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        mem_after_load, _ = get_gpu_metrics(handle)

        all_inference_times = []
        all_power_readings = []
        peak_mem_reading = mem_after_load

        print(f"Starting Monte Carlo simulation with {N} runs for model: {model_path}")
        
        for i in range(N):
            sentence = random.choice(monte_carlo_sentences)
            _, current_time = predict_sentiment_and_time(sentence, model, tokenizer, device)
            all_inference_times.append(current_time)

            mem_during_run, power_during_run = get_gpu_metrics(handle)
            all_power_readings.append(power_during_run)
            if mem_during_run > peak_mem_reading:
                peak_mem_reading = mem_during_run

            if (i + 1) % (N // 10) == 0:
                print(f"Completed {i + 1}/{N} runs.")
        
        average_inference_time = sum(all_inference_times) / len(all_inference_times)
        average_power_consumed = sum(all_power_readings) / len(all_power_readings) if all_power_readings else 0
        
        if len(all_inference_times) > 1:
            variance_time = sum((t - average_inference_time) ** 2 for t in all_inference_times) / (len(all_inference_times) - 1)
            std_dev_time = variance_time ** 0.5
        else:
            std_dev_time = 0.0

        model_results = {
            "model_name": model_path,
            "avg_inference_time_s": average_inference_time,
            "std_dev_time_s": std_dev_time,
            "peak_gpu_memory_mb": peak_mem_reading,
            "avg_gpu_power_w": average_power_consumed
        }
        results.append(model_results)

        print("=" * 60)
        print("Monte Carlo Results:")
        print(f"Model: {model_path}")
        print(f"Total Runs: {N}")
        print(f"Average Inference Time: {average_inference_time:.6f} seconds/sentence")
        print(f"Standard Deviation (Time): {std_dev_time:.6f} seconds")
        if gpu_available:
            print(f"Peak GPU Memory Used: {peak_mem_reading:.2f} MB")
            print(f"Average GPU Power Consumed: {average_power_consumed:.2f} W")
        print("=" * 60)

    # Save results to CSV
    csv_file = "results/benchmarks/inference_results.csv"
    csv_columns = ["model_name", "avg_inference_time_s", "std_dev_time_s", "peak_gpu_memory_mb", "avg_gpu_power_w"]
    
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in results:
                writer.writerow(data)
        print(f"Results successfully saved to {csv_file}")
    except IOError:
        print("I/O error")

# Shutdown NVML
if gpu_available:
    pynvml.nvmlShutdown()
