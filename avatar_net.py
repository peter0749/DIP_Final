import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_encoder import Encoder, Decoder, AdaIN
from style_decorator import StyleDecorator

class AvatarNet(nn.Module):
    def __init__(self):
        super(AvatarNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.decorater = StyleDecorator()

    def forward(self, content, styles, style_weight=1.0, patch_size=3, patch_stride=1, masks=None, interp_weights=None, train=False):
        if interp_weights is None:
            interp_weights = [1/len(styles)] * len(styles)
        
        if masks is None:
            masks = [1] * len(styles)
        
        cotent_feats = self.encoder(content)
        style_feats = []
        for style in styles:
            style_feats.append(self.encoder(style))
        
        trans_feats = []
        if train == True:
            trans_feats = cotent_feats[-1]
        else:
            for style_feat, interp_weight, mask in zip(style_feats, interp_weights, masks):
                if isinstance(mask, torch.Tensor):
                    _, _, h, w = cotent_feats[-1].size()
                    mask = F.interpolate(mask, size=(h, w))
                trans_feats.append(self.decorater(cotent_feats[-1], style_feat[-1], style_weight, patch_size, patch_stride) * interp_weight * mask)
            trans_feats = torch.sum(torch.stack(trans_feats, dim=0), dim=0)
        
        style_feats = [style_feat[:-1][::-1] for style_feat in style_feats]
        stylized_img = self.decoder(trans_feats, style_feats, masks, interp_weights)

        return stylized_img