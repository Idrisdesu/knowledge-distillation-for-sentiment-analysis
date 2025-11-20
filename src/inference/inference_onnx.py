# Libraries
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification 
from tqdm import tqdm
import time
import random
import pynvml
import csv
import os
from src.inference.inference import get_gpu_metrics, predict_sentiment_and_time, load_sentences_from_file

# NVML Initialization
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming GPU is at index 0
    gpu_available = True
except pynvml.NVMLError:
    gpu_available = False
    handle = None
    print("NVIDIA GPU not found. Running in CPU mode. GPU metrics will not be available.")


if __name__ == "__main__":
    models_to_benchmark = {
        "results/distilled_models_imdb_int8/distilled_distilbert_imdb_int8_ptq_onnx": 
            "results/distilled_model_imdb/distilled_distilbert_imdb",
        
        "results/distilled_models_imdb_int8/distilled_distilroberta_imdb_int8_ptq_onnx":
            "results/distilled_model_imdb/distilled_distilroberta_imdb",

        "results/distilled_models_imdb_int8/distilled_minilm_imdb_int8_ptq_onnx":
            "results/distilled_model_imdb/distilled_minilm_imdb",

        "results/distilled_models_imdb_int8/distilled_tinybert_imdb_int8_ptq_onnx":
            "results/distilled_model_imdb/distilled_tinybert_imdb"
    }

    monte_carlo_sentences = load_sentences_from_file("monte_carlo_sentences.txt")
    if not monte_carlo_sentences:
        print("No sentences loaded for Monte Carlo simulation. Exiting.")
        exit(1)

    N = 5000
    results = []

    for model_path, tokenizer_source_path in models_to_benchmark.items():

        if not os.path.isdir(model_path):
            print(f"Model directory not found: {model_path}. Skipping.")
            continue
        if not os.path.isdir(tokenizer_source_path):
            print(f"Tokenizer directory not found: {tokenizer_source_path}. Skipping.")
            continue

        print(f"\n--- Benchmarking ONNX Model: {model_path} ---")
        
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() and gpu_available else "CPUExecutionProvider"
        device = torch.device("cuda" if "CUDA" in provider else "cpu")
        print(f"Using ONNX Runtime provider: {provider}")

        # 2. Load ONNX model with ORTModelForSequenceClassification
        try:
            model = ORTModelForSequenceClassification.from_pretrained(
                model_path, 
                provider=provider,
                file_name="model_quantized.onnx"
            )
            print(f"ONNX model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading ONXX model from: {e}")
            continue

        # 3. Load the tokenizer from the original model
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_path)
            print(f"Tokenizer loaded from: {tokenizer_source_path}")
        except Exception as e:
            print(f"Error loading the tokenizer from {tokenizer_source_path}: {e}")
            continue
            
        if device.type == 'cuda' and gpu_available:
            torch.cuda.synchronize()

        mem_after_load, _ = (get_gpu_metrics(handle) if gpu_available else (0, 0))

        all_inference_times = []
        all_power_readings = []
        peak_mem_reading = mem_after_load

        print(f"Starting Monte Carlo simulation with {N} runs...")
        
        progress_bar = tqdm(range(N), desc="Benchmarking", leave=False)
        for i in progress_bar:
            sentence = random.choice(monte_carlo_sentences)
            _, current_time = predict_sentiment_and_time(sentence, model, tokenizer, device)
            all_inference_times.append(current_time)

            if gpu_available:
                mem_during_run, power_during_run = get_gpu_metrics(handle)
                all_power_readings.append(power_during_run)
                if mem_during_run > peak_mem_reading:
                    peak_mem_reading = mem_during_run

            if (i + 1) % (N // 100) == 0:
                 progress_bar.set_postfix({
                     'avg_time_ms': f'{(sum(all_inference_times) / (i + 1)) * 1000:.2f}'
                 })
        
        # Compute statistics
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

        # Print results for the current model
        print("\n" + "=" * 60)
        print("Monte Carlo Results:")
        print(f"Model: {model_path}")
        print(f"Total Runs: {N}")
        print(f"Average Inference Time: {average_inference_time:.6f} seconds/sentence")
        print(f"Standard Deviation (Time): {std_dev_time:.6f} seconds")
        if gpu_available:
            print(f"Peak GPU Memory Used: {peak_mem_reading:.2f} MB")
            print(f"Average GPU Power Consumed: {average_power_consumed:.2f} W")
        else:
            print("GPU metrics not available (CPU mode).")
        print("=" * 60)
    
    # Save results to CSV
    csv_file = "results/benchmarks/inference_results_quantized.csv"
    csv_columns = ["model_name", "avg_inference_time_s", "std_dev_time_s", "peak_gpu_memory_mb", "avg_gpu_power_w"]
    
    if results:
        try:
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in results:
                    writer.writerow(data)
            print(f"\nResults successfully saved to {csv_file}")
        except IOError:
            print("I/O error while saving CSV.")
    else:
        print("\nNo models were successfully benchmarked. CSV file not saved.")

    # Shutdown NVML
    if gpu_available:
        pynvml.nvmlShutdown()
