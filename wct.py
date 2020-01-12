import numpy as np
import torch

def covsqrt_mean(feature, inverse=False, tolerance=1e-10):
    b, c, h, w = feature.size()

    mean = torch.mean(feature.view(b, c, -1), dim=2, keepdim=True)
    zeromean = feature.view(b, c, -1) - mean
    cov = torch.bmm(zeromean, zeromean.transpose(1, 2))

    evals, evects = torch.symeig(cov, eigenvectors=True) # sort by evals: small -> big !!! BUGGY !!!
    evals_cpu = evals.detach().cpu()
    '''
    evals, evects = np.linalg.eig(cov.detach().cpu().numpy())
    ind = np.argsort(evals, axis=-1) # shape: (1, 512)
    evals = np.take_along_axis(evals, ind, axis=-1)
    evects = np.take_along_axis(evects, ind[np.newaxis], axis=-1)
    evals = torch.tensor(evals, dtype=feature.dtype, device=feature.device)
    evects = torch.tensor(evects, dtype=feature.dtype, device=feature.device)
    '''

    p = 0.5
    if inverse:
        p = -p

    covsqrt = []
    for i in range(b):
        l, r = -1, c-1 # (]
        while l<r-1:
            m = (l+r)//2
            if evals_cpu[i][m] > tolerance: # hit
                r = m # leftest hit
            else: # not hit
                l = m
        r = max(0,r)
        covsqrt.append(torch.mm(evects[i][:, r:],
                            torch.mm(evals[i][r:].pow(p).diag_embed(),
                                     evects[i][:, r:].t())).unsqueeze(0))
    covsqrt = torch.cat(covsqrt, dim=0)

    return covsqrt, mean


def whitening(feature):
    b, c, h, w = feature.size()

    inv_covsqrt, mean = covsqrt_mean(feature, inverse=True)

    normalized_feature = torch.matmul(inv_covsqrt, feature.view(b, c, -1)-mean)

    return normalized_feature.view(b, c, h, w)


def coloring(feature, target):
    b, c, h, w = feature.size()

    covsqrt, mean = covsqrt_mean(target)

    colored_feature = torch.matmul(covsqrt, feature.view(b, c, -1)) + mean

    return colored_feature.view(b, c, h, w)

