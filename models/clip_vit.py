import clip
import torch
import torch.nn as nn

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ImageEncoder(nn.Module):

    def __init__(self, device):
        super(ImageEncoder, self).__init__()
        self.device=device
        self.encoder=(clip.load("ViT-B/16", device=device))[0].visual 
    def forward(self, x):
        """
        Expects a tensor of size (batch_size, 3, 224, 224)
        """
        x = x.type(self.encoder.conv1.weight.dtype)
        x = self.encoder.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  
        x = self.encoder.transformer(x)
        grid_feats = x.permute(1, 0, 2)  
        grid_feats = self.encoder.ln_post(grid_feats[:,1:])  
        grid_feats_0 = self.encoder.ln_post(grid_feats[:,0])

        return grid_feats_0.float(), grid_feats.float()

    