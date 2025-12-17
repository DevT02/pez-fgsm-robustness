# FGSM Adversarial Robustness Testing for PEZ + CLIP

<img src=examples/teaser.png  width="70%" height="40%">

**Abstract:** This work evaluates the adversarial robustness of PEZ ([Wen et al., 2023](https://arxiv.org/abs/2302.03668)), a gradient-based discrete prompt optimization method for CLIP, under FGSM perturbations. We find that while PEZ-optimized prompts show some robustness to small adversarial noise, the discrete optimization process introduces significant variance that makes reliable evaluation challenging.

## Overview

Experimental robustness evaluation using FGSM perturbations on PEZ prompt optimization ([Wen et al., 2023](https://arxiv.org/abs/2302.03668)). Tests robustness of CLIP-based prompt learning under adversarial image noise.

**Tested:** Python 3.10, Windows 11, PyTorch 2.1+, CUDA 11.8 (RTX 3060)

## Setup

```bash
conda create -n fgsm-pez python=3.10
conda activate fgsm-pez

# GPU (PyTorch CUDA 13.0 wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# If that fails, use the PyTorch selector or try cu128/cu126
# See: https://pytorch.org/get-started/locally/

# CPU only
# pip install torch torchvision

# Install remaining dependencies
pip install -r requirements.txt

# Verify
python test_installation.py
```

## Quick Start

```bash
python pez_baseline.py assets/image.png --iterations 500
# Runtime: ~2-5 min (GPU), ~15-30 min (CPU)
```

Full evaluation:
```bash
python fgsm_evaluation.py assets/image.png --epsilon 0.01
python fgsm_evaluation.py assets/image.png --epsilon-range 0.001 0.02 0.005
```

## Configuration

`config.json`:
```json
{
  "prompt_len": 16,
  "iter": 1000,
  "lr": 0.1,
  "weight_decay": 0.1,
  "clip_model": "ViT-H-14",
  "clip_pretrain": "laion2b_s32b_b79k"
}
```

Use `ViT-B-32` / `openai` for lower memory.

## Results

Results saved to `results/` with complete metrics. Example findings:

| Epsilon | Embed Distance | Similarity Drop |
|---------|----------------|-----------------|
| 0.001   | 0.004          | -5.8%           |
| 0.021   | 0.070          | -9.1%           |
| 0.041   | 0.105          | +7.4%           |

**Metrics:**
- **Embed Distance**: L2 distance between clean and adversarial image embeddings in CLIP space
- **Similarity Drop**: Percent change in CLIP similarity score between optimized prompt and image (negative = degradation)

Larger perturbations increase embedding distance. High variance due to discrete optimization. See `results/README.md` for details.

## Method

1. PEZ optimizes discrete prompts via gradient-based search in CLIP embedding space
2. FGSM generates perturbations: $x' = x + \epsilon \cdot \mathrm{sign}(\nabla_x L)$
3. Compare prompt quality on clean vs perturbed images

## Limitations

- PEZ produces noisy prompts (discrete optimization, local minima)
- High run-to-run variance due to stochastic initialization
- Results measure relative robustness, not absolute prompt quality
- Small-norm perturbations may not align with human perception

Note: Optimization uses random initialization. For reproducible experiments, results should be averaged across multiple runs.

## Reproducibility

For deterministic results:
- Set `--seed` flag in evaluation scripts where available
- Run multiple trials (N≥5) and report mean±std
- Use same CLIP model/pretraining throughout experiments
- GPU/CPU differences may affect numerical precision

## Attribution

This implementation builds on:
- **PEZ algorithm**: [Hard Prompts Made Easy](https://github.com/YuxinWenRick/hard-prompts-made-easy) by Wen et al. (2023)
- **OpenCLIP**: Vendored snapshot from [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) (v2.23.0), unmodified except for local imports

*Note: Core development and experiments were conducted in 2023-2024. This repository represents a cleaned-up public release from December 2025.*

## Citation

```bibtex
@article{wen2023hard,
  title={Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery},
  author={Wen, Yuxin and Jain, Neel and Kirchenbauer, John and Goldblum, Micah and Geiping, Jonas and Goldstein, Tom},
  journal={arXiv preprint arXiv:2302.03668},
  year={2023}
}
```

## License

MIT
