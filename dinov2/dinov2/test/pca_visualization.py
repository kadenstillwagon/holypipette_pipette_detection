import torch
from torchvision import transforms as T
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2 # For resizing and interpolation
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import DinoVisionTransformer directly for un-sharded instantiation
from dinov2.dinov2.models.vision_transformer import vit_small, vit_large, vit_base
from dinov2.dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.dinov2.models import build_model_from_cfg
from dinov2.dinov2.data.transforms import CLAHETransform, make_normalize_transform


def run_pca_visualization_DINOPipetteDetection(model, img_size, fig_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_backbone_checkpoint = model.encoder.dino_encoder.state_dict()
    
    vit_model = vit_base(
        img_size=img_size, #896
        patch_size=14,
        block_chunks=0,
        num_register_tokens=4,
        init_values=1e-5
    ).to(device)
    vit_model.load_state_dict(teacher_backbone_checkpoint)

    img = Image.open(f'../datasets/test_image.png').convert('RGB')
    original_image_np = np.array(img)
    transform = T.Compose(
        [
            T.ToTensor(),
            make_normalize_transform(), # need to edit mean and std
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        ]
    )
    img = transform(img).unsqueeze(0).to(device)

    dino_out = vit_model(img)
    patch_features = dino_out['x_norm_patchtokens']

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_features[0].detach().cpu())


    pca_features_normalized = np.zeros_like(pca_features)
    for i in range(pca_features.shape[1]):
        min_val = pca_features[:, i].min()
        max_val = pca_features[:, i].max()
        if max_val - min_val > 0:
            pca_features_normalized[:, i] = (pca_features[:, i] - min_val) / (max_val - min_val)
        else:
            pca_features_normalized[:, i] = 0.5 # Handle case where component has no variance

    num_patches = int(img_size / 14)

    pca_image = pca_features_normalized.reshape(num_patches, num_patches, 3) # Reshape to a grid

    # 5. Upscale and Visualize
    # Upscale the PCA image to the original cropped image dimensions for visualization
    pca_image_upscaled = cv2.resize(pca_image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Display the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image_np)
    plt.title("Original Image (Cropped)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pca_image_upscaled)
    plt.title(f"DINOv2 PCA Visualization")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

    # Print explained variance ratio (how much variance each PC captures)
    print("\nExplained variance ratio by principal components:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.4f}")
    print(f"  Total explained by 3 PCs: {pca.explained_variance_ratio_.sum():.4f}")
