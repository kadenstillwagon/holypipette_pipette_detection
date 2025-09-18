import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from utils import LayerNorm2d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dinov2.dinov2.models.vision_transformer import vit_base

class DINOPipetteDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_dim = 768
        self.decoder_dim = 512
        feat_size=64
        self.img_size=518

        # Initialize dino encoder
        self.encoder = DINO_Encoder(self.img_size, self.embed_dim, self.decoder_dim)

        self.num_upsamples = int(math.sqrt(self.img_size // feat_size))

        self.feat_dim = self.decoder_dim // (2**(self.num_upsamples + 1))
        # Initialize prediction head
        self.prediction_head = PipetteDetectionPredictionHead(self.decoder_dim)

        #freeze required layers
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        x_encoded = self.encoder(x)

        preds = self.prediction_head(x_encoded)
        return preds

##################################
#            ENCODER
##################################

class DINO_Encoder(nn.Module):
    def __init__(self, img_size, embed_dim, decoder_dim):
        super().__init__()

        self.num_patches = int(img_size / 14)

        #create DINO base vit model
        self.dino_encoder = vit_base(
            img_size=img_size, #896
            patch_size=14,
            block_chunks=0,
            num_register_tokens=4,
            init_values=1e-5
        )

        self.vision_neck = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_dim, kernel_size=(1,1), bias=False),
            LayerNorm2d(decoder_dim),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=(3, 3), padding=1, bias=False),
            LayerNorm2d(decoder_dim)
        )

    def forward(self, x, **kwargs):
        # embed patches
        x = self.dino_encoder(x)['x_norm_patchtokens'].to(x.device)

        B, N, D = x.shape
        x = x.reshape(
            B, self.num_patches, self.num_patches, -1
        ).permute(0, 3, 1, 2) # Shape: (B, embed_dim, num_patches, num_patches)

        x = self.vision_neck(x).to(x.device)

        return x #BaseModelOutput(last_hidden_state=x)



##################################
#       PREDICTION HEADS
##################################

class PipetteDetectionPredictionHead(nn.Module):
    def __init__(self, decoder_dim):
        super().__init__()

        self.xy_prediction_head = PipetteXYCoordPredictionHead(decoder_dim)
        self.z_coord_prediction_head = PipetteZPredictionHead(decoder_dim)


    def forward(self, x_encoded):

        xy_head_input = x_encoded

        xy_pred = self.xy_prediction_head(xy_head_input)
        z_coord_pred = self.z_coord_prediction_head(x_encoded)
        return (xy_pred, z_coord_pred)


class PipetteXYCoordPredictionHead(nn.Module):
    def __init__(self, decoder_dim):
        super().__init__()

        self.pool_embeddings = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(decoder_dim, decoder_dim // 2),
            nn.ReLU(),
            nn.Linear(decoder_dim // 2, 2)
        )


    def forward(self, x_encoded):
        pooled_embedings = self.pool_embeddings(x_encoded)#.squeeze(2).squeeze(2)
        xy_pred = self.mlp(pooled_embedings).squeeze(1)

        return xy_pred


class PipetteZPredictionHead(nn.Module):
    def __init__(self, decoder_dim):
        super().__init__()

        self.pool_embeddings = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(decoder_dim, decoder_dim // 2),
            nn.ReLU(),
            nn.Linear(decoder_dim // 2, 1)
        )


    def forward(self, x_encoded):
        pooled_embedings = self.pool_embeddings(x_encoded)#.squeeze(2).squeeze(2)
        z_pred = self.mlp(pooled_embedings).squeeze(1)
        
        return z_pred