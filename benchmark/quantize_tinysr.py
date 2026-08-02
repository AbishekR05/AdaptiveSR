import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.modules.backends.fsrcnn_backend import load_model, FSRCNN_model
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    print("====================================================")
    print("Exporting FSRCNN (tinysr) to ONNX and Quantizing to INT8")
    print("====================================================\n")
    
    scale = 2
    os.makedirs("models/tinysr", exist_ok=True)
    
    # 1. Load the PyTorch model (this will auto-download weights if missing)
    print(f"Loading PyTorch FSRCNN model for scale={scale}...")
    pytorch_model = load_model("cpu", scale=scale)
    pytorch_model.eval()
    
    # 2. Define dummy input matching shape (B, C, H, W)
    # We use a standard 480p frame size (1, 3, 480, 640)
    dummy_input = torch.randn(1, 3, 480, 640)
    
    # 3. Export to ONNX
    onnx_path = f"models/tinysr/fsrcnn_x{scale}.onnx"
    print(f"Exporting PyTorch model to ONNX at: {onnx_path}...")
    
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    )
    print("ONNX model exported successfully!")
    
    # 4. Quantize ONNX to INT8
    quantized_path = f"models/tinysr/fsrcnn_x{scale}_int8.onnx"
    print(f"Quantizing ONNX model dynamically to INT8 at: {quantized_path}...")
    
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8
    )
    print("INT8 dynamic quantization completed successfully!")
    print(f"Models saved under: {os.path.abspath('models/tinysr')}")

if __name__ == "__main__":
    main()
