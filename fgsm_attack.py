#!/usr/bin/env python3
"""
FGSM attack implementation for CLIP.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class FGSMAttack:
    """FGSM attack against CLIP image encoder."""
    
    def __init__(self, model, epsilon=0.01, device="cpu"):
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.model.eval()
        
    def _preprocess_image(self, image, preprocess):
        if isinstance(image, Image.Image):
            img_tensor = preprocess(image).unsqueeze(0).to(self.device)
        else:
            img_tensor = image
        return img_tensor
    
    def _tensor_to_pil(self, tensor):
        # Denormalize from CLIP preprocessing
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(tensor.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(tensor.device)
        
        img = tensor.squeeze(0).clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        
        img = img.clamp(0, 1)
        img = img.mul(255).byte().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        return Image.fromarray(img)
    
    def generate_untargeted(self, image, preprocess):
        # Preprocess image
        img_tensor = self._preprocess_image(image, preprocess)
        img_tensor.requires_grad = True
        
        # Forward pass to get original embedding
        with torch.enable_grad():
            original_embedding = self.model.encode_image(img_tensor)
            original_embedding = F.normalize(original_embedding, dim=-1)
            
            # Loss: negative self-similarity (untargeted attack)
            # We want to maximize distance from original embedding
            loss = (original_embedding * original_embedding).sum()
            
        # Backward pass
        loss.backward()
        
        # Generate perturbation
        grad_sign = img_tensor.grad.sign()
        adv_tensor = img_tensor + self.epsilon * grad_sign
        
        # Clip to valid image range (after CLIP normalization)
        # CLIP normalizes to ~[-2.5, 2.5] range, but we clip conservatively
        adv_tensor = torch.clamp(adv_tensor, img_tensor - self.epsilon * 2, img_tensor + self.epsilon * 2)
        
        # Convert back to PIL
        adv_image = self._tensor_to_pil(adv_tensor.detach())
        
        return adv_image, img_tensor.detach(), adv_tensor.detach()
    
    def generate_targeted(self, image, target_text, preprocess, tokenizer):
        img_tensor = self._preprocess_image(image, preprocess)
        img_tensor.requires_grad = True
        
        # Get target text embedding
        with torch.no_grad():
            text_tokens = tokenizer([target_text]).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = F.normalize(text_embedding, dim=-1)
        
        # Forward pass
        with torch.enable_grad():
            img_embedding = self.model.encode_image(img_tensor)
            img_embedding = F.normalize(img_embedding, dim=-1)
            
            # Loss: negative similarity to target (minimize distance)
            loss = -(img_embedding * text_embedding).sum()
            
        # Backward pass
        loss.backward()
        
        # Generate perturbation (negative gradient for targeted attack)
        grad_sign = img_tensor.grad.sign()
        adv_tensor = img_tensor - self.epsilon * grad_sign  # Note: subtract for targeted
        
        # Clip to valid range
        adv_tensor = torch.clamp(adv_tensor, img_tensor - self.epsilon * 2, img_tensor + self.epsilon * 2)
        
        # Convert back to PIL
        adv_image = self._tensor_to_pil(adv_tensor.detach())
        
        return adv_image, img_tensor.detach(), adv_tensor.detach()
    
    def compute_embedding_distance(self, img1_tensor, img2_tensor):
        with torch.no_grad():
            emb1 = self.model.encode_image(img1_tensor)
            emb2 = self.model.encode_image(img2_tensor)
            
            emb1 = F.normalize(emb1, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)
            
            similarity = (emb1 * emb2).sum().item()
            distance = 1 - similarity
            
        return distance, similarity
    
    def compute_perturbation_stats(self, original_tensor, adv_tensor):
        diff = (adv_tensor - original_tensor).squeeze()
        
        stats = {
            "l0_norm": (diff.abs() > 1e-6).sum().item(),
            "l2_norm": diff.norm(p=2).item(),
            "linf_norm": diff.abs().max().item(),
            "mean_abs_diff": diff.abs().mean().item(),
        }
        
        return stats


def test_fgsm():
    import open_clip
    
    print("Testing FGSM Attack...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    # Create dummy image
    dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    # Initialize attack
    fgsm = FGSMAttack(model, epsilon=0.01, device=device)
    # Generate adversarial example
    adv_img, orig_tensor, adv_tensor = fgsm.generate_untargeted(dummy_img, preprocess)
    # Compute metrics
    distance, similarity = fgsm.compute_embedding_distance(orig_tensor, adv_tensor)
    stats = fgsm.compute_perturbation_stats(orig_tensor, adv_tensor)
    
    print(f"Embedding distance: {distance:.4f}")
    print(f"Embedding similarity: {similarity:.4f}")
    print(f"Perturbation stats: {stats}")
    print("Test passed!")


if __name__ == "__main__":
    test_fgsm()

