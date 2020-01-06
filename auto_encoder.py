import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.models

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg_feats = torchvision.models.vgg19(pretrained=True).features

        layer_index = [1, 6, 11, 20]

        self.layers = nn.ModuleList()
        tmp_seq = nn.Sequential()
        for i in range(layer_index[-1]+1):
            tmp_seq.add_module(str(i), vgg_feats[i])
            if i in layer_index:
                self.layers.append(tmp_seq)
                tmp_seq = nn.Sequential()

    def forward(self, x):
        out = x
        feats = []
        for layer in self.layers:
            out = layer(out)
            feats.append(out)
        
        return feats

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content, style, style_weight=1.0, eps=1e-5):
        n, c, h, w = content.size()

        content_std , content_mean = torch.std_mean(content.view(n, c, -1), dim=2, keepdim=True)
        style_std, style_mean = torch.std_mean(style.view(n, c, -1), dim=2, keepdim=True)

        norm_content = (content.view(n, c, -1)- content_mean) / (content_std + eps)
        stylized_content = (norm_content * style_std) + style_mean

        out = (1-style_weight) * content + style_weight * stylized_content.view(n, c, h, w)
        
        return out

class Decoder(nn.Module):
    def __init__(self, transforms=[AdaIN(), AdaIN(), AdaIN(), None]):
        super(Decoder, self).__init__()
        
        vgg_feats = torchvision.models.vgg19(pretrained=False).features
        self.transforms = transforms

        self.layers = nn.ModuleList()
        tmp_seq = nn.Sequential()

        layer_index = [1, 6, 11, 20]
        cnt = 0
        for i in range(max(layer_index)-1, -1, -1):
            if isinstance(vgg_feats[i], nn.Conv2d):
                out_channels = vgg_feats[i].in_channels
                in_channels = vgg_feats[i].out_channels
                kernel_size = vgg_feats[i].kernel_size

                tmp_seq.add_module(str(cnt), nn.ReflectionPad2d(padding=(1,1,1,1)))
                cnt += 1
                tmp_seq.add_module(str(cnt), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size))
                cnt += 1
                tmp_seq.add_module(str(cnt), nn.ReLU())
                cnt += 1
            
            elif isinstance(vgg_feats[i], nn.MaxPool2d):
                tmp_seq.add_module(str(cnt), nn.Upsample(scale_factor=2))
                cnt += 1
            
            if i in layer_index:
                self.layers.append(tmp_seq)
                tmp_seq = nn.Sequential()
        self.layers.append(tmp_seq[:-1])

    def forward(self, x, styles, masks=None, interp_weights=None):
        if interp_weights is None:
            interp_weights = [1/len(styles)] * len(styles)
        
        if masks is None:
            masks = [1] * len(styles)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)

            if self.transforms[i] is not None:
                trans_feats = []
                for style, interp_weight, mask in zip(styles, interp_weights, masks):
                    if isinstance(mask, torch.Tensor):
                        n, c, h, w = out.size()
                        mask = F.interpolate(mask, size=(h, w))
                    trans_feats.append(self.transforms[i](out, style[i])*interp_weight*mask)
                out = torch.sum(torch.stack(trans_feats, dim=0), dim=0)
        
        return out