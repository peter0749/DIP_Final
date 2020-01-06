import torch

def covsqrt_mean(feature, inverse=False, tolerance=1e-14):
    b, c, h, w = feature.size()

    mean = torch.mean(feature.view(b, c, -1), dim=2, keepdim=True)
    zeromean = feature.view(b, c, -1) - mean
    cov = torch.bmm(zeromean, zeromean.transpose(1, 2))

    evals, evects = torch.symeig(cov, eigenvectors=True) # sort by evals: small -> big

    p = 0.5
    if inverse:
        p = -p

    covsqrt = []
    for i in range(b):
        l, r = 0, len(c)-1 # must have at least 1 value (l=0); (]
        while l<r:
            m = (l+r)//2
            if evals[i][m] > tolerance: # hit
                r = m # leftest hit
            else: # not hit
                l = m
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

