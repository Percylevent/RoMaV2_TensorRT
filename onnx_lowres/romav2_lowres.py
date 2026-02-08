from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
from pathlib import Path

# Adjust path to find romav2 modules from parent directory
current_dir = Path(__file__).parent.absolute()
fast_dir = current_dir.parent.absolute()
sys.path.insert(0, str(fast_dir))
# Add dinov3 path
dino_dir = fast_dir / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
sys.path.insert(0, str(dino_dir))

from romav2.device import device
from romav2.features import Descriptor, FineFeatures
from romav2.geometry import bhwc_interpolate
from romav2.matcher import Matcher
from romav2.refiner import Refiners

def _interpolate_warp_and_confidence(
    *,
    warp: torch.Tensor,
    confidence: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
    zero_out_precision: bool,
):
    warp = bhwc_interpolate(
        warp,
        size=(H // patch_size, W // patch_size),
        mode="bilinear",
        align_corners=False,
    )
    if zero_out_precision:
        confidence[..., 1:] = 0.0

    confidence = bhwc_interpolate(
        confidence,
        size=(H // patch_size, W // patch_size),
        mode="bilinear",
        align_corners=False,
    )
    return warp, confidence

class RoMaV2LowRes(nn.Module):
    def __init__(self, weights_path: str, H_lr: int = 512, W_lr: int = 512, bidirectional: bool = True):
        super().__init__()
        
        # Configuration
        self.H_lr = H_lr
        self.W_lr = W_lr
        self.anchor_width = 512 # Keep anchor fixed to 512 as per model design
        self.anchor_height = 512
        self.bidirectional = bidirectional
        
        print(f"Initializing RoMaV2LowRes with: H={H_lr}, W={W_lr}, Bidirectional={bidirectional}")
        
        from romav2.refiner import ConvRefiner

        # Initialize sub-modules using default configs
        # Disable AMP for clean ONNX export (float32 only)
        self.f = Descriptor(Descriptor.Cfg(name="dinov3_vitl16", enable_amp=False))
        self.matcher = Matcher(Matcher.Cfg(style="romav2", enable_amp=False))
        
        # Configure refiners with AMP disabled manually
        self.refiners = nn.ModuleDict()
        
        # 4x refinement
        self.refiners["4"] = ConvRefiner(ConvRefiner.Cfg(
            feat_dim=256,
            proj_dim=192,
            displacement_emb_dim=79,
            local_corr_radius=3,
            patch_size=4,
            block_type="roma",
            enable_amp=False
        ))
        
        # 2x refinement
        self.refiners["2"] = ConvRefiner(ConvRefiner.Cfg(
            feat_dim=128,
            proj_dim=48,
            displacement_emb_dim=23,
            local_corr_radius=1,
            patch_size=2,
            block_type="roma",
            enable_amp=False
        ))
        
        # 1x refinement
        self.refiners["1"] = ConvRefiner(ConvRefiner.Cfg(
            feat_dim=64,
            proj_dim=12,
            displacement_emb_dim=8,
            local_corr_radius=None,
            patch_size=1,
            block_type="roma",
            enable_amp=False
        ))

        self.refiner_features = FineFeatures(FineFeatures.Cfg(type="vgg19bn"))
        
        # Load weights
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        self.load_state_dict(state_dict, strict=False) 
        self.eval()

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor):
        # Feature extraction
        f_A = self.f(img_A)
        f_B = self.f(img_B)
        
        # Match features (Use self.bidirectional set at init)
        matcher_output = self.matcher(
            f_A, f_B, img_A=img_A, img_B=img_B, bidirectional=self.bidirectional
        )
        
        warp_AB = matcher_output["warp_AB"]
        confidence_AB = matcher_output["confidence_AB"]
        warp_BA = matcher_output.get("warp_BA")
        confidence_BA = matcher_output.get("confidence_BA")

        if warp_BA is None:
            warp_BA = torch.zeros_like(warp_AB)
            confidence_BA = torch.zeros_like(confidence_AB)
        
        # Refinement (Single stage: Low-Res)
        # Using input shape dynamically or self.H_lr if fixed export
        B, C, H, W = img_A.shape 
        
        scale_factor = torch.tensor(
            (W / self.anchor_width, H / self.anchor_height), device=img_A.device
        )
        
        refiner_features_A = self.refiner_features(img_A)
        refiner_features_B = self.refiner_features(img_B)
        
        # Loop over patch sizes explicitly
        # The refiners are ordered by patch size in Refiners class logic
        # Here we manually iterate: 4 -> 2 -> 1
        for patch_size_str in ["4", "2", "1"]:
            refiner = self.refiners[patch_size_str]
            patch_size = int(patch_size_str)
            
            # Interpolate to current patch size resolution
            warp_AB, confidence_AB = _interpolate_warp_and_confidence(
                warp=warp_AB,
                confidence=confidence_AB,
                H=H,
                W=W,
                patch_size=patch_size,
                zero_out_precision=False, 
            )
            
            warp_BA, confidence_BA = _interpolate_warp_and_confidence(
                warp=warp_BA,
                confidence=confidence_BA,
                H=H,
                W=W,
                patch_size=patch_size,
                zero_out_precision=False,
            )

            f_patch_A = refiner_features_A[patch_size]
            f_patch_B = refiner_features_B[patch_size]
            
            # Refine AB
            refiner_output_AB = refiner(
                f_A=f_patch_A,
                f_B=f_patch_B,
                prev_warp=warp_AB,
                prev_confidence=confidence_AB,
                scale_factor=scale_factor,
            )
            warp_AB = refiner_output_AB["warp"]
            confidence_AB = refiner_output_AB["confidence"]
            
            # Refine BA (only if bidirectional is True? No, output shape must be consistent)
            # Even if bidirectional is False, Matcher outputs dummy BA, refine ensures consistency.
            refiner_output_BA = refiner(
                f_A=f_patch_B,
                f_B=f_patch_A,
                prev_warp=warp_BA,
                prev_confidence=confidence_BA,
                scale_factor=scale_factor,
            )
            warp_BA = refiner_output_BA["warp"]
            confidence_BA = refiner_output_BA["confidence"]
            
        return warp_AB, confidence_AB, warp_BA, confidence_BA
