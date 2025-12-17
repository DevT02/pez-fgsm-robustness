# Results Directory

Evaluation results from FGSM robustness testing.

## Sample Results

Testing PEZ robustness across epsilon values (ViT-B-32, 500 iterations):

| Epsilon | Embedding Distance | Similarity Drop | Drop % |
|---------|-------------------|-----------------|--------|
| 0.001   | 0.0037            | -0.0195         | -5.8%  |
| 0.011   | 0.0494            | -0.0001         | -0.0%  |
| 0.021   | 0.0699            | -0.0299         | -9.1%  |
| 0.031   | 0.0882            | +0.0018         | +0.5%  |
| 0.041   | 0.1047            | +0.0255         | +7.4%  |

**Observations:**
- Larger perturbations increase embedding distance
- No consistent degradation trend due to stochastic PEZ optimization
- Discrete token optimization introduces high variance across runs
- System demonstrates sensitivity to adversarial perturbations

## Files

- `fgsm_eval_YYYYMMDD_HHMMSS.json` - Complete metrics
- `*_adv_eps*.png` - Adversarial images (visually identical to original)

## Results Format

```json
{
  "input_image": "image.png",
  "epsilon_values": [0.01],
  "results": [{
    "epsilon": 0.01,
    "clean": {
      "prompt": "a photo of a cat",
      "clip_similarity": 0.8234
    },
    "adversarial": {
      "prompt": "blurry animal image",
      "clip_similarity": 0.7000
    },
    "comparison": {
      "similarity_drop": 0.1234,
      "similarity_drop_percent": 15.0
    },
    "perturbation": {
      "embedding_distance": 0.0234,
      "l2_norm": 0.0456,
      "linf_norm": 0.0100
    }
  }]
}
```

## Metrics

- **clip_similarity** - Cosine similarity between image and optimized prompt (higher is better)
- **similarity_drop** - Reduction in quality due to adversarial perturbation
- **embedding_distance** - Distance between clean and adversarial image in CLIP space
- **l2_norm** - L2 norm of perturbation
- **linf_norm** - L-infinity norm of perturbation
