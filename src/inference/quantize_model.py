from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import os
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime.configuration import CalibrationConfig, CalibrationMethod
'''
We need an ONNX ecosystem for PTQ quantization.
Thus, first we export the FP32 model to ONNX then we quantize it with a decicated calibration dataloader.
'''

# In the case of PTQ, we need a ONNX ecosystem for our models thus we need a calibration dataset along with a tokenizer and model
def get_calibration_dataloader(model_path, max_length=256, num_samples=100):
    print(f"Loading the tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading IMDb dataset for calibration...")
    ds = load_dataset("stanfordnlp/imdb")["train"].train_test_split(test_size=0.1, seed=42)["train"]
    calibration_ds = ds.shuffle(seed=42).select(range(100))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=384 # The same length as used during training
        )

    calibration_ds = calibration_ds.map(tokenize_function, batched=True)
    columns_to_remove = [col for col in ["text", "label", "labels"] if col in calibration_ds.column_names]
    calibration_ds = calibration_ds.remove_columns(columns_to_remove)
    return calibration_ds

def main_quantize(model_path):
    base_output_name = os.path.basename(model_path.rstrip('/\\'))
    onnx_fp32_path = f"{model_path}_onnx_fp32"
    output_dir = f"results/{base_output_name}_int8_ptq_onnx"
    
    print(f"Starting PTQ quantization for: {model_path}")
    print(f"Output located in: {output_dir}")

    print(f"\n[1/4] Loading dataset for calibration...")
    calibration_ds = get_calibration_dataloader(model_path, num_samples=100)

    print("\n[2/4] Calibration and quantization parameters...")
    calib_config = CalibrationConfig(
        dataset_name="stanfordnlp/imdb",
        dataset_config_name=None,
        dataset_split="train",
        dataset_num_samples=100,
        method=CalibrationMethod.MinMax
    )
    quant_config = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False)

    print("\n[3/4] Export PyTorch models towards ONNX...")
    try:
        model_fp32_onnx = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            export=True
        )
        model_fp32_onnx.save_pretrained(onnx_fp32_path)
        print(f"FP32 ONNX model saved in: {onnx_fp32_path}")
    except Exception as e:
        print(f"Error while exporting the model towards ONNX: {e}")
        return

    print(f"\n[4/4] Loading FP32 model for quantization...")
    quantizer = ORTQuantizer.from_pretrained(onnx_fp32_path)

    print("\n[5/5] Calibration (fit)...")
    
    calibration_ranges = None 
    try:
        calibration_ranges = quantizer.fit(
            dataset=calibration_ds,
            calibration_config=calib_config,
        )
        print("Calibration completed.")
    except Exception as e:
        print(f"Error while fitting: {e}")
        return

    print("[5/5] INT8 quantization...")
    try:
        quantizer.quantize(
            save_dir=output_dir,
            quantization_config=quant_config,
            calibration_tensors_range=calibration_ranges
        )
        print(f"Quantization finished.")
        print(f"INT8 model saved in : {output_dir}")
    except Exception as e:
        print(f"Error while .quantize(): {e}")
        return

    # Inference test
    print("Inference test...")
    try:
        model_int8 = ORTModelForSequenceClassification.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        text = "This movie was surprisingly good and emotional."
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model_int8(**inputs)
        prediction = outputs.logits.argmax().item()
        label_map = {0: "negative", 1: "positive"}
        print(f"Input: {text}")
        print(f"INT8 prediction: {label_map.get(prediction, 'unknown')}")
        print("Inference test completed successfully.")
    except Exception as e:
        print(f"Error while inference test: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTQ quantization of a distilled student model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned student model directory."
    )
    args = parser.parse_args()
    if not os.path.isdir(args.model_path):
        print(f"Error: the folder '{args.model_path}' doesn't exist.")
    else:
        main_quantize(args.model_path)
