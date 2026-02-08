from __future__ import annotations
print("SCRIPT START", flush=True)

import argparse
import sys
import torch
import torch.onnx
from pathlib import Path

# Setup paths
print("Setting up paths...", flush=True)
current_dir = Path(__file__).parent.absolute()
fast_dir = current_dir.parent.absolute()
dino_dir = fast_dir / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
sys.path.insert(0, str(dino_dir))
sys.path.insert(0, str(fast_dir))
sys.path.insert(0, str(current_dir)) 

print("Importing RoMaV2LowRes...", flush=True)
from romav2_lowres import RoMaV2LowRes
print("Importing device...", flush=True)
from romav2.device import device
print("Imports done.", flush=True)

def export_onnx(size: int, bidirectional: bool):
    # Model configuration
    H_lr = size
    W_lr = size
    
    # Load Model
    weights_path = fast_dir / "romav2.pt"
    print(f"Loading Low-Res model from {weights_path}")
    print(f"Configuration: Size={size}x{size}, Bidirectional={bidirectional}")
    
    model = RoMaV2LowRes(
        str(weights_path), 
        H_lr=H_lr, 
        W_lr=W_lr, 
        bidirectional=bidirectional
    )
    model.eval()
    model.cuda()

    # Create dummy input with dynamic axes in mind, but fixed for export tracing
    # We use the requested size for the dummy input
    B, C = 1, 3
    img_A = torch.randn(B, C, H_lr, W_lr).cuda()
    img_B = torch.randn(B, C, H_lr, W_lr).cuda()
    
    # Define output name based on config
    bi_str = "bi" if bidirectional else "uni"
    output_filename = f"romav2_{size}_{bi_str}.onnx"
    output_path = current_dir / output_filename
    
    print(f"Exporting to {output_path}...")
    
    # Wrap in autocast(enabled=False) just to be super safe, though model internals force it too
    with torch.autocast(device_type="cuda", enabled=False):
        torch.onnx.export(
            model,
            (img_A, img_B),
            str(output_path),
            input_names=["img_A", "img_B"],
            output_names=["warp_AB", "confidence_AB", "warp_BA", "confidence_BA"],
            opset_version=17,
            do_constant_folding=False, # Safe default
            dynamic_axes={
                "img_A": {0: "batch_size"},
                "img_B": {0: "batch_size"},
                "warp_AB": {0: "batch_size"},
                "confidence_AB": {0: "batch_size"},
                "warp_BA": {0: "batch_size"},
                "confidence_BA": {0: "batch_size"},
            }
        )
    print("Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RoMaV2 Low-Res to ONNX")
    parser.add_argument("--size", type=int, default=512, help="Input resolution (both H and W), default: 512")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="Enable bidirectional matching (default: True)")
    parser.add_argument("--no-bidirectional", action="store_false", dest="bidirectional", help="Disable bidirectional matching")
    
    args = parser.parse_args()
    
    export_onnx(size=args.size, bidirectional=args.bidirectional)
