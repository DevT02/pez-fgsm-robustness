#!/usr/bin/env python3
"""
Test installation.
"""

import sys
mport torch

def test_imports():
    """Test that all required packages can be imported."""
    print("\nImports:")
    
    tests = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("open_clip", "OpenCLIP"),
    ]
    
    failed = []
    
    for module_name, display_name in tests:
        try:
            __import__(module_name)
            print(f"  [OK] {display_name}")
        except (ImportError, AttributeError) as e:
            print(f"  [FAIL] {display_name}: {str(e)[:60]}")
            failed.append(display_name)
    
    return len(failed) == 0, failed


def test_torch_cuda():
    """Test PyTorch CUDA availability."""
    print("\nPyTorch:")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU devices: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.get_device_name(0)}")
    else:
        print("  Running on CPU")
        print("  For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
    
    return True


def test_open_clip():
    """Test OpenCLIP model loading."""
    print("\nOpenCLIP:")
    
    try:
        import open_clip
        
        # Try loading a small model
        print("  Loading ViT-B-32 model...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            device="cpu"  # Use CPU for testing
        )
        print("  [OK] Model loaded successfully")
        
        # Test tokenizer
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer(["a test sentence"])
        print(f"  [OK] Tokenizer working (output shape: {tokens.shape})")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_fgsm_module():
    """Test FGSM attack module."""
    print("\nFGSM:")
    
    try:
        from fgsm_attack import FGSMAttack
        print("  [OK] FGSM module imported")
        
        import torch
        import open_clip
        from PIL import Image
        
        # Load minimal model
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device="cpu"
        )
        
        # Create dummy image
        dummy_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        
        # Test attack
        fgsm = FGSMAttack(model, epsilon=0.01, device="cpu")
        adv_img, orig_tensor, adv_tensor = fgsm.generate_untargeted(dummy_img, preprocess)
        
        print("  [OK] FGSM attack generated successfully")
        
        # Compute metrics
        distance, similarity = fgsm.compute_embedding_distance(orig_tensor, adv_tensor)
        print(f"  [OK] Embedding distance: {distance:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optim_utils():
    """Test optimization utilities."""
    print("\nConfig:")
    
    try:
        from optim_utils import read_json
        
        # Test config loading
        config = read_json("config.json")
        print(f"  [OK] Config loaded: {config.get('clip_model', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing installation...")
    
    results = []
    
    # Test imports
    success, failed = test_imports()
    results.append(("Imports", success))
    
    if not success:
        print(f"\nFailed: {', '.join(failed)}")
        print("Create fresh environment:")
        print("  conda create -n fgsm-pez python=3.10")
        print("  conda activate fgsm-pez")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Test PyTorch
    results.append(("PyTorch", test_torch_cuda()))
    
    # Test OpenCLIP
    results.append(("OpenCLIP", test_open_clip()))
    
    # Test optimization utils
    results.append(("Optim Utils", test_optim_utils()))
    
    # Test FGSM module
    results.append(("FGSM Module", test_fgsm_module()))
    
    # Summary
    all_passed = True
    for test_name, passed in results:
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed.")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


