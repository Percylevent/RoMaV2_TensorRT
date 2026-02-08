# RoMaV2 Flexible ONNX Export (in onnx_lowres)

This directory contains scripts to export RoMaV2 models with customizable resolution and matching modes.

## Supported Modes
*   **Fast**: Resolution 512, Bidirectional=True
*   **Base**: Resolution 640, Bidirectional=False
*   **Turbo**: Resolution 320, Bidirectional=False
*   **Custom**: Any resolution and bidirectional setting.

## Files
*   **`romav2_lowres.py`**: The model definition supporting `H_lr`, `W_lr`, and `bidirectional` init arguments.
*   **`export_lowres_onnx.py`**: Export script with CLI args.
*   **`verify_lowres_onnx.py`**: Verification script with CLI args.

## Usage

### 1. Exporting

**Note**: When exporting with `--no-bidirectional`, the model will still output 4 tensors (`warp_AB`, `confidence_AB`, `warp_BA`, `confidence_BA`) to maintain a consistent signature. However, `warp_BA` and `confidence_BA` will be filled with zeros.

**Export Fast Mode (Default 512, Bi)**:
```bash
python export_lowres_onnx.py --size 512 --bidirectional
```
Output: `romav2_512_bi.onnx`

**Export Base Mode (640, Uni)**:
```bash
python export_lowres_onnx.py --size 640 --no-bidirectional
```
Output: `romav2_640_uni.onnx`

**Export Turbo Mode (320, Uni)**:
```bash
python export_lowres_onnx.py --size 320 --no-bidirectional
```
Output: `romav2_320_uni.onnx`

### 2. Verifying

Use the same arguments as export:

```bash
python verify_lowres_onnx.py --size 512 --bidirectional
python verify_lowres_onnx.py --size 640 --no-bidirectional
```

### 3. Visualizing

Use the same arguments as export to visualize results:

```bash
python visualize_lowres_onnx.py --size 512 --bidirectional
python visualize_lowres_onnx.py --size 640 --no-bidirectional
```

Output images will be saved as `vis_{size}_{bi/uni}.jpg` in the current directory.
