from __future__ import annotations

import argparse
import sys
import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent.absolute()
fast_dir = current_dir.parent.absolute()
dino_dir = fast_dir / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
sys.path.insert(0, str(dino_dir))
sys.path.insert(0, str(fast_dir))
sys.path.insert(0, str(current_dir)) 

from romav2_highres import RoMaV2HighRes
from romav2.device import device

def verify_onnx(size_L: int, size_H: int, bidirectional: bool):
    H_lr = size_L
    W_lr = size_L
    H_hr = size_H
    W_hr = size_H
    
    bi_str = "bi" if bidirectional else "uni"
    onnx_filename = f"romav2_L{size_L}_H{size_H}_{bi_str}.onnx"
    onnx_path = current_dir / onnx_filename
    
    print(f"Verifying {onnx_filename}...")
    
    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found. Please run export first.")
        return

    # 1. Run PyTorch Model for Ground Truth
    print("Running PyTorch model...")
    weights_path = fast_dir / "romav2.pt"
    
    model = RoMaV2HighRes(
        str(weights_path), 
        H_lr=H_lr, 
        W_lr=W_lr,
        H_hr=H_hr,
        W_hr=W_hr,
        bidirectional=bidirectional
    )
    model.cuda()
    model.eval()
    
    # Dummy Input
    B, C = 1, 3
    torch.manual_seed(42)
    img_A_lr = torch.randn(B, C, H_lr, W_lr).cuda()
    img_B_lr = torch.randn(B, C, H_lr, W_lr).cuda()
    img_A_hr = torch.randn(B, C, H_hr, W_hr).cuda()
    img_B_hr = torch.randn(B, C, H_hr, W_hr).cuda()
    
    with torch.inference_mode():
        pt_warp_AB, pt_conf_AB, pt_warp_BA, pt_conf_BA = model(img_A_lr, img_B_lr, img_A_hr, img_B_hr)
        
    # 2. Run ONNX Model
    print("Running ONNX model...")
    
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return

    ort_inputs = {
        "img_A_lr": img_A_lr.cpu().numpy(),
        "img_B_lr": img_B_lr.cpu().numpy(),
        "img_A_hr": img_A_hr.cpu().numpy(),
        "img_B_hr": img_B_hr.cpu().numpy(),
    }
    
    ort_outs = session.run(None, ort_inputs)
    onnx_warp_AB = ort_outs[0]
    onnx_conf_AB = ort_outs[1]
    onnx_warp_BA = ort_outs[2]
    onnx_conf_BA = ort_outs[3]
    
    # 3. Compare
    print("Comparing outputs...")
    
    def check_diff(name, pt_tensor, onnx_array, tol=1e-3):
        pt_np = pt_tensor.cpu().numpy()
        diff = np.abs(pt_np - onnx_array)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"{name}: Max Diff = {max_diff:.6f}, Mean Diff = {mean_diff:.6f}")
        if max_diff > tol:
            print(f"  WARNING: {name} difference exceeds tolerance {tol}")
            return False
        return True

    ok = True
    ok &= check_diff("warp_AB", pt_warp_AB, onnx_warp_AB)
    ok &= check_diff("confidence_AB", pt_conf_AB, onnx_conf_AB, tol=0.1) 
    ok &= check_diff("warp_BA", pt_warp_BA, onnx_warp_BA)
    ok &= check_diff("confidence_BA", pt_conf_BA, onnx_conf_BA, tol=0.1)
    
    if ok:
        print("VERIFICATION PASSED!")
    else:
        print("VERIFICATION FAILED: Significant differences found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify RoMaV2 High-Res ONNX")
    parser.add_argument("--sizeL", type=int, default=512, help="Low-res input resolution, default: 512")
    parser.add_argument("--sizeH", type=int, default=768, help="High-res input resolution, default: 768")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="Enable bidirectional matching (default: True)")
    parser.add_argument("--no-bidirectional", action="store_false", dest="bidirectional", help="Disable bidirectional matching")
    
    args = parser.parse_args()
    
    verify_onnx(size_L=args.sizeL, size_H=args.sizeH, bidirectional=args.bidirectional)
