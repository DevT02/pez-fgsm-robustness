#!/usr/bin/env python3
"""
PEZ baseline - optimize prompts from images using CLIP.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
import open_clip
from optim_utils import optimize_prompt, read_json

def parse_args():
    parser = argparse.ArgumentParser(
        description="PEZ: Find optimal text prompts from images using CLIP"
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=str,
        help="Path(s) to input image(s). Multiple images optimize a single prompt across all."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration JSON file (default: config.json)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of optimization iterations (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save optimized prompt to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress during optimization"
    )
    return parser.parse_args()

def load_images(image_paths):
    """Load and validate input images."""
    images = []
    for path in image_paths:
        if not Path(path).exists():
            print(f"Error: Image not found: {path}")
            sys.exit(1)
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            print(f"Loaded: {path} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            sys.exit(1)
    return images

def main():
    args = parse_args()
    
    # Load images
    images = load_images(args.images)
    print(f"Loaded {len(args.images)} image(s)")
    
    # Load configuration
    try:
        config = argparse.Namespace()
        config.__dict__.update(read_json(args.config))
        
        # Override with command-line arguments
        if args.iterations is not None:
            config.iter = args.iterations
        if args.verbose:
            config.print_new_best = True
            if config.print_step is None:
                config.print_step = 100
        
        print(f"Config: {config.clip_model}, {config.iter} iters, lr={config.lr}")
        
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Creating default config.json...")
        import json
        default_config = {
            "prompt_len": 16,
            "iter": 1000,
            "lr": 0.1,
            "weight_decay": 0.1,
            "prompt_bs": 1,
            "loss_weight": 1.0,
            "print_step": 100,
            "batch_size": 1,
            "clip_model": "ViT-H-14",
            "clip_pretrain": "laion2b_s32b_b79k"
        }
        with open("config.json", "w") as f:
            json.dump(default_config, f, indent=4)
        print("Created config.json with defaults. Please run again.")
        sys.exit(1)
    
    # Initialize CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.clip_model, 
            pretrained=config.clip_pretrain, 
            device=device
        )
        model.eval()
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        sys.exit(1)
    
    # Optimize prompt
    print("Optimizing...")
    
    try:
        learned_prompt = optimize_prompt(
            model=model,
            preprocess=preprocess,
            args=config,
            device=device,
            target_images=images,
            target_prompts=None
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\nResult: \"{learned_prompt}\"")
    
    # Save output if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(learned_prompt)
        print(f"Saved to {args.output}")
    
    return learned_prompt

if __name__ == "__main__":
    main()

