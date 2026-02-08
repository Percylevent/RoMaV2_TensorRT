# RoMaV2 ONNX Export

This directory contains the files and scripts necessary to export, verify, and use the RoMaV2 model in ONNX format.

## Files Description

*   **`romav2_simplified.py`**: Contains the `RoMaV2Simplified` class. This is a stripped-down version of the RoMaV2 model optimized for "fast" mode (512x512 resolution, bidirectional matching). It has been modified to remove unused branches and ensure compatibility with ONNX export (e.g., forcing float32 precision).
*   **`export_fast_onnx.py`**: The script to export the simplified model to ONNX. It loads `RoMaV2Simplified` weights from `../romav2.pt`, creates dummy inputs, and exports the model to `romav2_fast.onnx`.
*   **`verify_fast_onnx.py`**: A script to verify the numerical correctness of the exported ONNX model. It compares the output of `romav2_fast.onnx` (using `onnxruntime-gpu`) against the PyTorch `RoMaV2Simplified` model on random inputs.
*   **`visualize_onnx.py`**: A script to visually verify the ONNX model. It runs inference on sample images (e.g., `../1.jpg`, `../2.jpg`) and saves the visualization to `onnx_visual_output.jpg`.
*   **`romav2_fast.onnx`**: The resulting ONNX model file (opset 17).

## Dependencies

*   `torch`
*   `onnx`
*   `onnxruntime-gpu`
*   `numpy`
*   `Pillow`
*   `einops`

## Usage

### 1. Prerequisite
Ensure you have the model weights `romav2.pt` in the parent directory (`../romav2.pt`) and the `facebookresearch_dinov3...` submodule available.

### 2. Exporting the Model
Run the export script:
```bash
python export_fast_onnx.py
```
This will generate `romav2_fast.onnx` in this directory.

### 3. Verifying the Model
To check numerical correctness:
```bash
python verify_fast_onnx.py
```

To visually check on images:
```bash
python visualize_onnx.py
```
Check `onnx_visual_output.jpg` for the result.

## Implementation Details

*   **Precision**: The model is exported in **float32**.
*   **Patches**:
    *   **RoPE**: The implementation of Rotary Positional Embeddings in `../romav2/vit/rope.py` was patched to strictly use `float32` to avoid `ComplexDouble` errors during ONNX export.
    *   **Autocast**: Hardcoded `torch.autocast` contexts in `../romav2/features.py` (VGG) and `../romav2/dpt.py` (DPTHead) were disabled/removed to prevent `bfloat16` tensors in the graph, which are incompatible with some ONNX runtimes or specific export configurations.
    *   **Constant Folding**: Disabled (`do_constant_folding=False`) during export to improve stability.

## Input/Output Spec

*   **Inputs**:
    *   `img_A`: `[B, 3, 512, 512]` (float32)
    *   `img_B`: `[B, 3, 512, 512]` (float32)
*   **Outputs**:
    *   `warp_AB`: `[B, 512, 512, 2]` - Warp from A to B
    *   `confidence_AB`: `[B, 512, 512, 1]` - Confidence for A to B
    *   `warp_BA`: `[B, 512, 512, 2]` - Warp from B to A
    *   `confidence_BA`: `[B, 512, 512, 1]` - Confidence for B to A
