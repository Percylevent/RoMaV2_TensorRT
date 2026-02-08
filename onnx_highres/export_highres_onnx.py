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

print("Importing RoMaV2HighRes...", flush=True)
from romav2_highres import RoMaV2HighRes
print("Importing device...", flush=True)
from romav2.device import device
print("Imports done.", flush=True)

def export_onnx(size_L: int, size_H: int, bidirectional: bool):
    # Model configuration
    H_lr = size_L
    W_lr = size_L
    H_hr = size_H
    W_hr = size_H
    
    # Load Model
    weights_path = fast_dir / "romav2.pt"
    print(f"Loading High-Res model from {weights_path}")
    print(f"Configuration: LR={size_L}x{size_L}, HR={size_H}x{size_H}, Bidirectional={bidirectional}")
    
    model = RoMaV2HighRes(
        str(weights_path), 
        H_lr=H_lr, 
        W_lr=W_lr,
        H_hr=H_hr,
        W_hr=W_hr,
        bidirectional=bidirectional
    )
    model.eval()
    model.cuda()

    # Create dummy inputs
    # All 4 inputs must be provided for ONNX export
    B, C = 1, 3
    img_A_lr = torch.randn(B, C, H_lr, W_lr).cuda()
    img_B_lr = torch.randn(B, C, H_lr, W_lr).cuda()
    img_A_hr = torch.randn(B, C, H_hr, W_hr).cuda()
    img_B_hr = torch.randn(B, C, H_hr, W_hr).cuda()
    
    # Define output name based on config
    bi_str = "bi" if bidirectional else "uni"
    output_filename = f"romav2_L{size_L}_H{size_H}_{bi_str}.onnx"
    output_path = current_dir / output_filename
    
    print(f"Exporting to {output_path}...")
    
    # Wrap in autocast(enabled=False) for safety
    with torch.autocast(device_type="cuda", enabled=False):
        torch.onnx.export(
            model,
            (img_A_lr, img_B_lr, img_A_hr, img_B_hr),
            str(output_path),
            input_names=["img_A_lr", "img_B_lr", "img_A_hr", "img_B_hr"],
            output_names=["warp_AB", "confidence_AB", "warp_BA", "confidence_BA"],
            opset_version=17,
            do_constant_folding=False,
            dynamic_axes={
                "img_A_lr": {0: "batch_size"},
                "img_B_lr": {0: "batch_size"},
                "img_A_hr": {0: "batch_size"},
                "img_B_hr": {0: "batch_size"},
                "warp_AB": {0: "batch_size"},
                "confidence_AB": {0: "batch_size"},
                "warp_BA": {0: "batch_size"},
                "confidence_BA": {0: "batch_size"},
            }
        )
    print("Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RoMaV2 High-Res to ONNX")
    parser.add_argument("--sizeL", type=int, default=512, help="Low-res input resolution (both H and W), default: 512")
    parser.add_argument("--sizeH", type=int, default=768, help="High-res input resolution (both H and W), default: 768")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="Enable bidirectional matching (default: True)")
    parser.add_argument("--no-bidirectional", action="store_false", dest="bidirectional", help="Disable bidirectional matching")
    
    args = parser.parse_args()
    
    export_onnx(size_L=args.sizeL, size_H=args.sizeH, bidirectional=args.bidirectional)
