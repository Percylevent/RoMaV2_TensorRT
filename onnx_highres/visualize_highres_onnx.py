from __future__ import annotations

import argparse
import sys
import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

# Setup paths
current_dir = Path(__file__).parent.absolute()
fast_dir = current_dir.parent.absolute()
dino_dir = fast_dir / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
sys.path.insert(0, str(dino_dir))
sys.path.insert(0, str(fast_dir))
sys.path.insert(0, str(current_dir)) 

from romav2_highres import RoMaV2HighRes
from romav2.device import device

def visualize_onnx(size_L: int, size_H: int, bidirectional: bool):
    H_lr = size_L
    W_lr = size_L
    H_hr = size_H
    W_hr = size_H
    
    bi_str = "bi" if bidirectional else "uni"
    onnx_filename = f"romav2_L{size_L}_H{size_H}_{bi_str}.onnx"
    onnx_path = current_dir / onnx_filename
    
    print(f"Visualizing {onnx_filename}...")
    
    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found. Please run export first.")
        return

    # Use sample images from workspace
    img_A_path = fast_dir / "a1.jpg"
    img_B_path = fast_dir / "a2.jpg"

    if not img_A_path.exists():
        img_A_path = fast_dir / "1.jpg"
    if not img_B_path.exists():
        img_B_path = fast_dir / "2.jpg"
    
    if not img_A_path.exists() or not img_B_path.exists():
        print(f"Error: Sample images not found at {img_A_path} or {img_B_path}")
        return

    print(f"Loading ONNX model from {onnx_path}...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return

    # Preprocess images
    print("Preprocessing images...")
    im1_pil = Image.open(img_A_path).convert("RGB")
    im2_pil = Image.open(img_B_path).convert("RGB")
    
    # Resize to LR and HR resolutions
    im1_pil_lr = im1_pil.resize((W_lr, H_lr), Image.BICUBIC)
    im2_pil_lr = im2_pil.resize((W_lr, H_lr), Image.BICUBIC)
    im1_pil_hr = im1_pil.resize((W_hr, H_hr), Image.BICUBIC)
    im2_pil_hr = im2_pil.resize((W_hr, H_hr), Image.BICUBIC)
    
    im1_lr_np = np.array(im1_pil_lr).transpose(2, 0, 1).astype(np.float32) / 255.0
    im2_lr_np = np.array(im2_pil_lr).transpose(2, 0, 1).astype(np.float32) / 255.0
    im1_hr_np = np.array(im1_pil_hr).transpose(2, 0, 1).astype(np.float32) / 255.0
    im2_hr_np = np.array(im2_pil_hr).transpose(2, 0, 1).astype(np.float32) / 255.0
    
    im1_lr = im1_lr_np[None]  # Add batch dim
    im2_lr = im2_lr_np[None]
    im1_hr = im1_hr_np[None]
    im2_hr = im2_hr_np[None]

    # Inference
    print("Running inference...")
    inputs = {
        "img_A_lr": im1_lr,
        "img_B_lr": im2_lr,
        "img_A_hr": im1_hr,
        "img_B_hr": im2_hr,
    }
    outs = session.run(None, inputs)
    
    # Parse outputs (high-res results)
    warp_AB = torch.from_numpy(outs[0])[0].cuda()
    confidence_AB = torch.from_numpy(outs[1])[0].cuda()
    warp_BA = torch.from_numpy(outs[2])[0].cuda()
    confidence_BA = torch.from_numpy(outs[3])[0].cuda()

    # Visualization logic matching demo_fast.py
    print("Visualizing results...")
    overlap_AB = confidence_AB[..., 0].sigmoid()
    overlap_BA = confidence_BA[..., 0].sigmoid()

    # Use high-res images for visualization
    im1_tensor = torch.from_numpy(im1_hr).cuda()
    im2_tensor = torch.from_numpy(im2_hr).cuda()
    
    x1 = im1_tensor[0]
    x2 = im2_tensor[0]

    im2_transfer_rgb = F.grid_sample(
        x2[None], warp_AB[None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
        x1[None], warp_BA[None], mode="bilinear", align_corners=False
    )[0]

    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)
    overlap = torch.cat((overlap_AB, overlap_BA), dim=1)
    white_im = torch.ones((H_hr, 2 * W_hr), device='cuda')
    vis_im = overlap * warp_im + (1 - overlap) * white_im

    # Save
    out_path = current_dir / f"vis_L{size_L}_H{size_H}_{bi_str}.jpg"
    from romav2.io import tensor_to_pil
    tensor_to_pil(vis_im).save(out_path)
    print(f"Result saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RoMaV2 High-Res ONNX")
    parser.add_argument("--sizeL", type=int, default=512, help="Low-res input resolution, default: 512")
    parser.add_argument("--sizeH", type=int, default=768, help="High-res input resolution, default: 768")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="Enable bidirectional matching (default: True)")
    parser.add_argument("--no-bidirectional", action="store_false", dest="bidirectional", help="Disable bidirectional matching")
    
    args = parser.parse_args()
    
    visualize_onnx(size_L=args.sizeL, size_H=args.sizeH, bidirectional=args.bidirectional)
