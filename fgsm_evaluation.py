#!/usr/bin/env python3
"""
FGSM robustness evaluation for PEZ.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
import open_clip
import json
from datetime import datetime

from optim_utils import optimize_prompt, read_json
from fgsm_attack import FGSMAttack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PEZ robustness against FGSM attacks"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="FGSM perturbation magnitude (default: 0.01)"
    )
    parser.add_argument(
        "--epsilon-range",
        nargs=3,
        type=float,
        metavar=("START", "END", "STEP"),
        help="Test multiple epsilon values (start, end, step)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration JSON (default: config.json)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of PEZ iterations (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/)"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save adversarial images to output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    return parser.parse_args()


def evaluate_single_epsilon(image, epsilon, model, preprocess, config, device, verbose=False):
    """
    Run full evaluation pipeline for a single epsilon value.
    
    Returns:
        dict with results
    """
    results = {
        "epsilon": epsilon,
        "clean": {},
        "adversarial": {},
        "comparison": {}
    }
    
    # Initialize FGSM attack
    fgsm = FGSMAttack(model, epsilon=epsilon, device=device)
    
    # Generate adversarial example
    if verbose:
        print(f"Generating FGSM (eps={epsilon})...")
    
    adv_image, orig_tensor, adv_tensor = fgsm.generate_untargeted(image, preprocess)
    
    # Compute perturbation statistics
    distance, similarity = fgsm.compute_embedding_distance(orig_tensor, adv_tensor)
    perturb_stats = fgsm.compute_perturbation_stats(orig_tensor, adv_tensor)
    
    results["perturbation"] = {
        "embedding_distance": distance,
        "embedding_similarity": similarity,
        **perturb_stats
    }
    
    if verbose:
        print(f"Embed dist: {distance:.4f}, Linf: {perturb_stats['linf_norm']:.6f}")
    
    # Run PEZ on clean image
    if verbose:
        print("Running PEZ on clean image...")
    
    try:
        clean_prompt = optimize_prompt(
            model=model,
            preprocess=preprocess,
            args=config,
            device=device,
            target_images=[image],
            target_prompts=None
        )
        results["clean"]["prompt"] = clean_prompt
        results["clean"]["success"] = True
        
        # Compute CLIP similarity for clean prompt
        with torch.no_grad():
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            img_features = model.encode_image(img_tensor)
            
            tokenizer = open_clip.get_tokenizer(config.clip_model)
            text_tokens = tokenizer([clean_prompt]).to(device)
            text_features = model.encode_text(text_tokens)
            
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            clean_similarity = (img_features @ text_features.T).item()
            results["clean"]["clip_similarity"] = clean_similarity
        
        if verbose:
            print(f"Prompt: \"{clean_prompt}\", sim: {clean_similarity:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
        results["clean"]["success"] = False
        results["clean"]["error"] = str(e)
        return results
    
    # Run PEZ on adversarial image
    if verbose:
        print("Running PEZ on adversarial image...")
    
    try:
        adv_prompt = optimize_prompt(
            model=model,
            preprocess=preprocess,
            args=config,
            device=device,
            target_images=[adv_image],
            target_prompts=None
        )
        results["adversarial"]["prompt"] = adv_prompt
        results["adversarial"]["success"] = True
        
        # Compute CLIP similarity for adversarial prompt
        with torch.no_grad():
            adv_img_tensor = preprocess(adv_image).unsqueeze(0).to(device)
            adv_img_features = model.encode_image(adv_img_tensor)
            
            text_tokens = tokenizer([adv_prompt]).to(device)
            adv_text_features = model.encode_text(text_tokens)
            
            adv_img_features = adv_img_features / adv_img_features.norm(dim=-1, keepdim=True)
            adv_text_features = adv_text_features / adv_text_features.norm(dim=-1, keepdim=True)
            
            adv_similarity = (adv_img_features @ adv_text_features.T).item()
            results["adversarial"]["clip_similarity"] = adv_similarity
        
        if verbose:
            print(f"Prompt: \"{adv_prompt}\", sim: {adv_similarity:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
        results["adversarial"]["success"] = False
        results["adversarial"]["error"] = str(e)
        return results
    
    # Comparison metrics
    if results["clean"]["success"] and results["adversarial"]["success"]:
        results["comparison"] = {
            "prompt_changed": clean_prompt != adv_prompt,
            "similarity_drop": clean_similarity - adv_similarity,
            "similarity_drop_percent": (clean_similarity - adv_similarity) / clean_similarity * 100
        }
        
        if verbose:
            print(f"Match: {not results['comparison']['prompt_changed']}, "
                  f"drop: {results['comparison']['similarity_drop']:.4f}")
    
    return results, adv_image


def main():
    args = parse_args()
    
    # Load image
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    image = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image} ({image.size[0]}x{image.size[1]})")
    
    # Load config
    try:
        config = argparse.Namespace()
        config.__dict__.update(read_json(args.config))
        if args.iterations is not None:
            config.iter = args.iterations
        if not args.verbose:
            config.print_step = None
            config.print_new_best = False
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Config: {config.clip_model}, {config.iter} iters, device={device}")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_model,
        pretrained=config.clip_pretrain,
        device=device
    )
    model.eval()
    
    # Determine epsilon values to test
    if args.epsilon_range:
        import numpy as np
        epsilon_values = np.arange(args.epsilon_range[0], args.epsilon_range[1], args.epsilon_range[2])
        epsilon_values = [float(e) for e in epsilon_values]
    else:
        epsilon_values = [args.epsilon]
    
    print(f"Testing epsilon: {epsilon_values}")
    
    # Run evaluations
    all_results = []
    adversarial_images = []
    
    for i, epsilon in enumerate(epsilon_values, 1):
        print(f"\n[{i}/{len(epsilon_values)}] epsilon={epsilon}")
        
        results, adv_img = evaluate_single_epsilon(
            image, epsilon, model, preprocess, config, device, verbose=args.verbose
        )
        
        all_results.append(results)
        adversarial_images.append((epsilon, adv_img))
        
        # Print summary
        if results["clean"]["success"] and results["adversarial"]["success"]:
            print(f"Clean: \"{results['clean']['prompt']}\"")
            print(f"Adv:   \"{results['adversarial']['prompt']}\"")
            print(f"Drop:  {results['comparison']['similarity_drop']:.4f} ({results['comparison']['similarity_drop_percent']:.1f}%)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"fgsm_eval_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "input_image": str(args.image),
            "config": config.__dict__,
            "epsilon_values": epsilon_values,
            "results": all_results
        }, f, indent=2)
    
    print(f"\nSaved: {results_file}")
    
    # Save adversarial images if requested
    if args.save_images:
        for epsilon, adv_img in adversarial_images:
            img_name = Path(args.image).stem
            adv_img_path = output_dir / f"{img_name}_adv_eps{epsilon:.4f}.png"
            adv_img.save(adv_img_path)
            print(f"Saved: {adv_img_path}")
    
    # Print final summary table
    if len(all_results) > 1:
        print(f"\n{'Epsilon':<12} {'Embed Dist':<12} {'Sim Drop':<12} {'Drop %':<12}")
        
        for res in all_results:
            if res["clean"]["success"] and res["adversarial"]["success"]:
                print(f"{res['epsilon']:<12.4f} "
                      f"{res['perturbation']['embedding_distance']:<12.4f} "
                      f"{res['comparison']['similarity_drop']:<12.4f} "
                      f"{res['comparison']['similarity_drop_percent']:<12.1f}")
    
    print()


if __name__ == "__main__":
    main()

