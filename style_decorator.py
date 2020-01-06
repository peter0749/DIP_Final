import torch
import torch.nn.functional as F
from wct import whitening, coloring

def extract_patches(feature, patch_size, stride):
    kh, kw = patch_size
    dh, dw = stride

    # padding input
    ph = int((kh-1)/2)
    pw = int((kw-1)/2)
    padding_size = (pw, pw, ph, ph)

    feature = F.pad(feature, padding_size, mode='reflect')
    kernels = feature.unfold(2, kh, dh).unfold(3, kw, dw)
    kernels = kernels.contiguous().view(*kernels.size()[:-2], -1)

    return kernels


class StyleDecorator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(StyleDecorator, self).__init__(**kwargs)
        self.eps = 1e-5


    def kernel_normalize(self, kernel, k=3):
        b, ch, h, w, kk = kernel.size()

        kernel = kernel.view(b, ch, h*w, kk).transpose(2, 1)
        kernel_norm = torch.norm(kernel.contiguous().view(b, h*w, ch*kk), p=2, dim=2, keepdim=True) + self.eps

        kernel = kernel.view(b, h*w, ch, k, k)
        kernel_norm = kernel_norm.view(b, h*w, 1, 1, 1)

        return kernel, kernel_norm


    def conv2d_with_style_kernels(self, features, kernels, patch_size, deconv_flag=False):
        output = []
        b, c, h, w = features.size()

        pad_size2 = patch_size-1
        pad_size = pad_size2//2
        padding_size = (pad_size,)*4

        for feature, kernel in zip(features, kernels):
            feature = feature.unsqueeze(0)
            feature = F.pad(feature, padding_size, mode='reflect')
            output.append(F.conv_transpose2d(feature, kernel, padding=pad_size2) if deconv_flag else F.conv2d(feature, kernel))

        return torch.cat(output, dim=0)


    def binarization_patch_score(self, features):
        outputs = []

        # batch-wise binarization
        for feature in features:
            # best matching patch index
            b, h, w = feature.size()
            matching_indices = torch.argmax(feature, dim=0)
            matching_indices_ravel = matching_indices.view(-1)
            one_hot_mask = torch.zeros((b, h*w), dtype=feature.dtype, device=feature.device).scatter_(0, matching_indices_ravel.unsqueeze(0), 1).view(b, h, w)
            outputs.append(one_hot_mask.unsqueeze(0))

        return torch.cat(outputs, dim=0)


    def norm_deconvolution(self, h, w, patch_size):
        mask = torch.ones((h, w))
        fullmask = torch.zeros( (h+patch_size-1, w+patch_size-1)  )
        for x in range(patch_size):
            for y in range(patch_size):
                paddings = (x, patch_size-x-1, y, patch_size-y-1)
                padded_mask = F.pad(mask, paddings, 'constant', 0)
                fullmask += padded_mask
        pad_width = (patch_size-1)//2
        if pad_width == 0:
            deconv_norm = fullmask
        else:
            deconv_norm = fullmask[pad_width:-pad_width, pad_width:-pad_width]
        return deconv_norm.view(1, 1, h, w)


    def reassemble_feature(self, normalized_content_feature, normalized_style_feature, patch_size, patch_stride):
        style_kernel = extract_patches(normalized_style_feature, [patch_size, patch_size],
                                                                 [patch_stride, patch_stride])
        style_kernel, kernel_norm = self.kernel_normalize(style_kernel, patch_size)
        patch_score = self.conv2d_with_style_kernels(normalized_content_feature, style_kernel/kernel_norm, patch_size)
        binarized = self.binarization_patch_score(patch_score)
        deconv_norm = self.norm_deconvolution(h=binarized.size(2), w=binarized.size(3), patch_size= patch_size)
        output = self.conv2d_with_style_kernels(binarized, style_kernel, patch_size, deconv_flag=True)

        return output/deconv_norm.type_as(output)


    def forward(self, content_feature, style_feature, style_strength=1.0, patch_size=3, patch_stride=1):
        normalized_content_feature = whitening(content_feature)
        normalized_style_feature = whitening(style_feature)
        reassembled_feature = self.reassemble_feature(normalized_content_feature, normalized_style_feature, patch_size=patch_size, patch_stride=patch_stride)
        stylized_feature = coloring(reassembled_feature, style_feature)
        result_feature = (1.-style_strength) * content_feature + style_strength * stylized_feature

        return result_feature

