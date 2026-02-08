# RoMaV2 Fast Inference & ONNX Export

This directory contains optimized implementations and ONNX export tools for the RoMaV2 (Robust Matching V2) model.

## Directory Structure

```
fast/
├── romav2/                    # Core RoMaV2 modules
├── facebookresearch_dinov3_.../  # DINOv3 backbone dependency
├── romav2.pt                  # Pre-trained weights (required)
├── onnx/                      # Fixed-configuration ONNX export
│   ├── romav2_simplified.py   # Simplified model for export
│   ├── export_fast_onnx.py    # Export script (512x512, bidirectional)
│   ├── verify_fast_onnx.py    # Numerical verification
│   └── visualize_onnx.py      # Visual verification
├── onnx_lowres/               # Flexible-configuration ONNX export
│   ├── romav2_lowres.py       # Flexible resolution model
│   ├── export_lowres_onnx.py  # Export with CLI args
│   ├── verify_lowres_onnx.py  # Numerical verification
│   └── visualize_lowres_onnx.py # Visual verification
├── requirements.txt           # Python dependencies
└── .gitignore                 # Excludes generated files
```

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download Weights**:
Place `romav2.pt` in the `fast/` directory. You can obtain it from the original [RoMaV2 repository](https://github.com/Parskatt/RoMaV2).

## Usage

### Option 1: Fixed Configuration (fast/onnx/)

Export RoMaV2 in "Fast" mode (512x512, bidirectional):

```bash
cd onnx
python export_fast_onnx.py
python verify_fast_onnx.py
python visualize_onnx.py
```

**Output**: `romav2_fast.onnx`

### Option 2: Flexible Configuration (fast/onnx_lowres/)

Export with custom resolution and matching mode:

```bash
cd onnx_lowres

# Fast mode (512x512, bidirectional)
python export_lowres_onnx.py --size 512 --bidirectional

# Base mode (640x640, unidirectional)
python export_lowres_onnx.py --size 640 --no-bidirectional

# Turbo mode (320x320, unidirectional)
python export_lowres_onnx.py --size 320 --no-bidirectional

# Verify and visualize
python verify_lowres_onnx.py --size 512 --bidirectional
python visualize_lowres_onnx.py --size 512 --bidirectional
```

**Outputs**: `romav2_{size}_{bi/uni}.onnx`, `vis_{size}_{bi/uni}.jpg`

## Technical Details

### Model Architecture

RoMaV2 performs dense correspondence matching between image pairs:
- **Feature Extraction**: DINOv3-ViT-L/16 backbone
- **Matcher**: Cross-attention based dense matching
- **Refinement**: Multi-scale convolutional refinement (4x → 2x → 1x)

### ONNX Export Notes

- **Precision**: All exports use `float32` (AMP disabled for ONNX compatibility)
- **Inputs**: `img_A`, `img_B` (float32, shape: `[B, 3, H, W]`, normalized to [0, 1])
- **Outputs**:
  - `warp_AB`: Dense flow field from A to B (shape: `[B, H, W, 2]`)
  - `confidence_AB`: Matching confidence (shape: `[B, H, W, C]`)
  - `warp_BA`: Dense flow field from B to A (zeros if unidirectional)
  - `confidence_BA`: Matching confidence (zeros if unidirectional)

### Portability

All scripts use **relative paths** based on `Path(__file__)`, ensuring the `fast/` directory can be moved or cloned to any location without modification.

## Troubleshooting

**Q: Import errors for `romav2` or `dinov3` modules?**  
A: Ensure you run scripts from their respective directories (`onnx/` or `onnx_lowres/`). The path setup is automatic.

**Q: ONNX export hangs or fails?**  
A: Ensure you have sufficient GPU memory. The model requires ~8GB VRAM for 512x512 inputs.

**Q: Verification shows large differences?**  
A: Small numerical differences (<1e-3 for warps, <0.1 for confidences) are expected due to ONNX runtime optimizations.

## Citation

If you use this code, please cite the original RoMaV2 paper:

```bibtex
@inproceedings{edstedt2024romav2,
  title={RoMa: Robust Dense Feature Matching},
  author={Edstedt, Johan and Sun, Qiyu and Bökman, Georg and Wadenb{\"a}ck, Mårten and Felsberg, Michael},
  booktitle={CVPR},
  year={2024}
}
```

## License

Follow the license terms of the original RoMaV2 repository.
