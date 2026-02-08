# RoMaV2 High-Resolution ONNX Export

This directory contains scripts to export RoMaV2 models with **multi-resolution support** (low-res + high-res refinement).

## Overview

Unlike `onnx_lowres/` which only processes low-resolution images, `onnx_highres/` supports the "precise" mode from the original RoMaV2 model, which performs two-stage refinement:
1. **Stage 0 (Low-Res)**: Initial matching and refinement at lower resolution
2. **Stage 1 (High-Res)**: Additional refinement at higher resolution for improved accuracy

## Model Inputs

The exported ONNX model requires **4 inputs** (all required):
- `img_A_lr`: Low-resolution image A (float32, shape: `[B, 3, H_lr, W_lr]`)
- `img_B_lr`: Low-resolution image B (float32, shape: `[B, 3, H_lr, W_lr]`)
- `img_A_hr`: High-resolution image A (float32, shape: `[B, 3, H_hr, W_hr]`)
- `img_B_hr`: High-resolution image B (float32, shape: `[B, 3, H_hr, W_hr]`)

## Usage

### 1. Exporting

**Note**: When exporting with `--no-bidirectional`, the model will still output 4 tensors, but `warp_BA` and `confidence_BA` will be filled with zeros.

**Export Precise Mode (Default: LR=512, HR=768, Bi)**:
```bash
python export_highres_onnx.py --sizeL 512 --sizeH 768 --bidirectional
```

**Export Custom Configuration**:
```bash
python export_highres_onnx.py --sizeL 640 --sizeH 960 --bidirectional
```

**Output**: `romav2_L{sizeL}_H{sizeH}_{bi/uni}.onnx`

### 2. Verification

Use the same arguments as export to verify numerical correctness:

```bash
python verify_highres_onnx.py --sizeL 512 --sizeH 768 --bidirectional
```

### 3. Visualizing

Use the same arguments as export to visualize results:

```bash
python visualize_highres_onnx.py --sizeL 512 --sizeH 768 --bidirectional
```

Output images will be saved as `vis_L{sizeL}_H{sizeH}_{bi/uni}.jpg` in the current directory.

## Technical Details

- **Two-stage refinement**: The model first processes low-res images, then refines using high-res images
- **Zero-out precision**: At the start of high-res refinement (stage 1, patch size 4), precision channels are zeroed out as per the original implementation
- **Float32 precision**: All exports use float32 (AMP disabled for ONNX compatibility)
- **Output resolution**: Final outputs match the high-resolution input size

## Comparison with onnx_lowres/

| Feature | onnx_lowres | onnx_highres |
|---------|-------------|--------------|
| Inputs | 2 (img_A, img_B) | 4 (img_A_lr, img_B_lr, img_A_hr, img_B_hr) |
| Refinement Stages | 1 (single resolution) | 2 (low-res → high-res) |
| Accuracy | Good | Better (more precise) |
| Speed | Faster | Slower (two passes) |
| Use Case | Fast inference | High-precision matching |
